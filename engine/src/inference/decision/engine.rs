//! The decision model's llama.cpp context: owned by one worker thread and
//! kept from one request to the next.
//!
//! A `LlamaContext` borrows its model and is not `Send`, so a context that
//! outlives a request has to live on a thread of its own: requests reach it
//! as [`Job`]s over a channel, one at a time. Between requests the worker
//! keeps two things. The context itself, created at the size the largest
//! request so far needed and grown when one needs more — so a request no
//! longer pays for creating one. And what sequence 0 holds: the shared part
//! of the last request, usually the state. A request whose shared part is
//! token for token the same starts from it instead of decoding it again,
//! which is what an agent asking several rounds of questions about one
//! document does.
//!
//! Neither changes an answer. A question lands in the same cells, in batches
//! of the same shape, over the same attention window whether the state was
//! decoded now or by an earlier request, and whatever the size of the
//! context: llama.cpp attends over the used cells rounded up to 256, which
//! a context sized in whole steps of that never caps.

use std::num::NonZeroU32;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Instant;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;

use super::{DecisionError, EvalMode, EvalStats, KV_WINDOW_STEP, ms_since, runtime, score_rows};

/// Fewest cells a context is created with: below this, growing it again on
/// the next slightly longer request would cost more than the memory saved.
pub(super) const MIN_CELLS: usize = 2048;

/// How prompts are cut into decode calls. Fixed per model: it is part of
/// what an answer is computed with, and a context is created for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Protocol {
    /// Most tokens one `llama_decode` call receives. A multiple of
    /// `ubatch`, so splitting a prompt into calls never moves a micro-batch
    /// boundary.
    pub batch: u32,
    /// llama.cpp's micro-batch: the unit the model's arithmetic is done in.
    pub ubatch: u32,
    /// Block-causal attention: a non-causal context fed one renderer block
    /// per decode call, so each block sees the cache (every earlier block)
    /// and all of itself.
    pub blocks: bool,
    /// Most questions one `batched` decode round holds, one sequence each.
    /// A recurrent layer keeps a state per sequence, so a hybrid model's
    /// rounds are kept small.
    pub group: usize,
    /// Flash attention off on the CPU whatever `--no-flash-attn` says: the
    /// model's published scores were validated that way.
    pub cpu_flash_attn_off: bool,
}

impl Protocol {
    /// The code readout: a causal chat prompt, llama.cpp's own micro-batch.
    pub const CODES: Self = Self {
        batch: 2048,
        ubatch: 512,
        blocks: false,
        group: super::MAX_QUESTIONS,
        cpu_flash_attn_off: false,
    };
}

/// What is read at each read position.
pub(super) enum Readout {
    /// The full-vocabulary log-probability of each class, `classes[prompt]`
    /// listing every class's tokens.
    Classes(Vec<Vec<Vec<LlamaToken>>>),
    /// `logit(yes) - logit(no)`.
    Verdict { yes: LlamaToken, no: LlamaToken },
}

/// One request's evaluation.
pub(super) struct Job {
    pub prompts: Vec<Vec<LlamaToken>>,
    /// Per prompt, the positions whose outputs are read, ascending, all at
    /// or after `shared`.
    pub reads: Vec<Vec<usize>>,
    /// Tokens at the start of `prompts[0]` that every prompt shares and that
    /// are decoded once, in sequence 0.
    pub shared: usize,
    /// Block-causal protocols: per prompt, the renderer blocks `[a, b)` that
    /// tile it, those of the shared part first. Empty otherwise.
    pub blocks: Vec<Spans>,
    pub readout: Readout,
    pub mode: EvalMode,
    /// F32 KV cache and no flash attention (`DecisionModel::exact`).
    pub exact: bool,
}

/// Per prompt: the class log-probabilities (`Readout::Classes`) or one
/// score per read position (`Readout::Verdict`).
pub(super) type JobResult = Result<(Vec<Vec<f64>>, EvalStats), DecisionError>;

pub(super) struct EngineConfig {
    pub threads: u32,
    /// `--no-flash-attn` not given.
    pub flash_attn: bool,
    /// Most cells one request may use (`--decision-ctx`).
    pub max_ctx: u32,
    /// The model runs on a GPU.
    pub on_gpu: bool,
    pub protocol: Protocol,
}

enum Message {
    Evaluate(Box<Job>, mpsc::SyncSender<JobResult>),
    Release(mpsc::SyncSender<()>),
}

/// The handle `DecisionModel` keeps: dropping it stops the worker and frees
/// the context before returning.
pub(super) struct Engine {
    tx: Mutex<Option<mpsc::Sender<Message>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Engine {
    pub fn start(
        model: Arc<LlamaModel>,
        backend: Arc<LlamaBackend>,
        config: EngineConfig,
    ) -> std::io::Result<Self> {
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("eullm-decision".into())
            .spawn(move || Worker::new(&model, &backend, &config).run(&rx))?;
        Ok(Self {
            tx: Mutex::new(Some(tx)),
            worker: Mutex::new(Some(worker)),
        })
    }

    pub fn evaluate(&self, job: Job) -> JobResult {
        let (reply, answer) = mpsc::sync_channel(1);
        self.send(Message::Evaluate(Box::new(job), reply))?;
        answer
            .recv()
            .map_err(|_| DecisionError::Runtime("the decision worker stopped".into()))?
    }

    /// Free the context, and with it the VRAM it holds, until the next
    /// request needs one: done before a generation model is sized, which
    /// counts the decision model's context as reserved, not as used.
    pub fn release(&self) {
        let (reply, done) = mpsc::sync_channel(1);
        if self.send(Message::Release(reply)).is_ok() {
            let _ = done.recv();
        }
    }

    fn send(&self, message: Message) -> Result<(), DecisionError> {
        self.tx
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .ok_or_else(|| DecisionError::Runtime("the decision worker has stopped".into()))?
            .send(message)
            .map_err(|_| DecisionError::Runtime("the decision worker has stopped".into()))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        drop(
            self.tx
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take(),
        );
        if let Some(worker) = self
            .worker
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            let _ = worker.join();
        }
    }
}

/// The context the worker keeps between requests.
struct Cached<'m> {
    ctx: LlamaContext<'m>,
    exact: bool,
    cells: usize,
    n_seq: u32,
    n_outputs: u32,
    /// What sequence 0 holds from position 0 and the blocks it was decoded
    /// in (none for a causal protocol), by `prefix`; `None` when it holds
    /// anything else.
    prefix: Option<(Vec<LlamaToken>, Spans)>,
}

/// Token ranges `[a, b)`, each decoded in a call of its own.
pub(super) type Spans = Vec<(usize, usize)>;

/// What one read position produced.
enum Read {
    Row(Vec<f32>),
    Score(f64),
}

struct Worker<'m> {
    model: &'m LlamaModel,
    backend: &'m LlamaBackend,
    config: &'m EngineConfig,
    cached: Option<Cached<'m>>,
}

impl<'m> Worker<'m> {
    fn new(model: &'m LlamaModel, backend: &'m LlamaBackend, config: &'m EngineConfig) -> Self {
        Self {
            model,
            backend,
            config,
            cached: None,
        }
    }

    fn run(mut self, rx: &mpsc::Receiver<Message>) {
        while let Ok(message) = rx.recv() {
            match message {
                Message::Evaluate(job, reply) => {
                    let result = match catch_unwind(AssertUnwindSafe(|| self.evaluate(&job))) {
                        Ok(result) => result,
                        Err(panic) => {
                            // Whatever the context holds now is not known.
                            self.cached = None;
                            let what = panic
                                .downcast_ref::<String>()
                                .map(String::as_str)
                                .or_else(|| panic.downcast_ref::<&str>().copied())
                                .unwrap_or("unknown panic");
                            Err(DecisionError::Runtime(format!(
                                "decision evaluation failed: {what}"
                            )))
                        }
                    };
                    let _ = reply.send(result);
                }
                Message::Release(done) => {
                    self.cached = None;
                    let _ = done.send(());
                }
            }
        }
    }

    fn evaluate(&mut self, job: &Job) -> JobResult {
        let blocks = self.config.protocol.blocks;
        match job.mode {
            EvalMode::Separate if blocks => self.separate_blocks(job),
            // Block attention decodes one block per call, so questions
            // cannot share one: `batched` is answered as `shared_prefix`,
            // and reported as such.
            _ if blocks => self.isolated_blocks(job),
            EvalMode::SharedPrefix => self.isolated(job),
            EvalMode::Batched if job.prompts.len() > 1 => self.batched(job),
            // One prompt has nothing to batch with.
            EvalMode::Batched => {
                let (scores, mut stats) = self.separate(job)?;
                stats.mode = EvalMode::Batched;
                Ok((scores, stats))
            }
            EvalMode::Separate => self.separate(job),
        }
    }

    fn check_fits(&self, needed: usize) -> Result<(), DecisionError> {
        let limit = self.config.max_ctx as usize;
        if needed > limit {
            return Err(DecisionError::TooLong { needed, limit });
        }
        Ok(())
    }

    /// The kept context if it is large enough for `cells`, `n_seq` and
    /// `n_outputs`, a new, larger one otherwise. Returns the milliseconds
    /// spent creating one (0 when kept).
    fn context(
        &mut self,
        exact: bool,
        cells: usize,
        n_seq: u32,
        n_outputs: u32,
    ) -> Result<f64, DecisionError> {
        // llama.cpp reserves room for `max(outputs, sequences)` rows and
        // aborts the process — not an error, an abort — when that exceeds
        // `n_outputs_max`.
        let n_outputs = n_outputs.max(n_seq);
        if let Some(c) = &self.cached
            && c.exact == exact
            && c.cells >= cells
            && c.n_seq >= n_seq
            && c.n_outputs >= n_outputs
        {
            return Ok(0.0);
        }
        let started = Instant::now();
        let step = KV_WINDOW_STEP.max(self.config.protocol.ubatch as usize);
        let (cells, n_seq, n_outputs) = match self.cached.take() {
            // Grow, never shrink: a request the old one fitted fits again.
            Some(old) if old.exact == exact => (
                cells.max(old.cells),
                n_seq.max(old.n_seq),
                n_outputs.max(old.n_outputs),
            ),
            _ => (cells, n_seq, n_outputs),
        };
        let protocol = self.config.protocol;
        // At least one decode call's worth, so llama.cpp never caps the
        // batch below the size prompts are cut into.
        let cells = cells
            .max(MIN_CELLS)
            .max(protocol.batch as usize)
            .next_multiple_of(step);
        let flash_attn = self.config.flash_attn
            && !exact
            && (self.config.on_gpu || !protocol.cpu_flash_attn_off);
        let threads = self.config.threads as i32;
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(u32::try_from(cells).unwrap_or(u32::MAX)))
            .with_n_batch(protocol.batch)
            .with_n_ubatch(protocol.ubatch)
            .with_n_seq_max(n_seq)
            .with_n_outputs_max(n_outputs)
            // Unified, so sequences share the prefix's cells: without it
            // llama.cpp gives each sequence its own slice and copying the
            // prefix would copy the data.
            .with_kv_unified(true)
            .with_non_causal_attention(protocol.blocks)
            .with_n_threads(threads)
            .with_n_threads_batch(threads)
            // Set either way, as `build_ctx_params_with_cache` does: left
            // alone, llama.cpp's default is AUTO, not off.
            .with_flash_attention_policy(if flash_attn { -1 } else { 0 });
        let params = if exact {
            params
                .with_type_k(KvCacheType::F32)
                .with_type_v(KvCacheType::F32)
        } else {
            params
        };
        let ctx = self.model.new_context(self.backend, params).map_err(|e| {
            DecisionError::Runtime(format!("Failed to create decision context: {e}"))
        })?;
        self.cached = Some(Cached {
            ctx,
            exact,
            cells,
            n_seq,
            n_outputs,
            prefix: None,
        });
        Ok(ms_since(started))
    }

    fn cached(&mut self) -> &mut Cached<'m> {
        self.cached
            .as_mut()
            .expect("context() creates the context before it is used")
    }

    /// Make sequence 0 hold exactly `prefix` — decoded in `blocks` for a
    /// block protocol, in batch-sized calls otherwise — and nothing else in
    /// the cache. Kept as it is when it already does. Returns the
    /// milliseconds spent and whether it was kept.
    fn prefix(
        &mut self,
        prefix: &[LlamaToken],
        blocks: &[(usize, usize)],
    ) -> Result<(f64, bool), DecisionError> {
        let started = Instant::now();
        let batch_size = self.config.protocol.batch as usize;
        let c = self.cached();
        if !prefix.is_empty()
            && c.prefix
                .as_ref()
                .is_some_and(|(tokens, b)| tokens == prefix && b == blocks)
        {
            return Ok((0.0, true));
        }
        c.prefix = None;
        clear(&mut c.ctx)?;
        if prefix.is_empty() {
            return Ok((ms_since(started), false));
        }
        let spans: Spans = if blocks.is_empty() {
            (0..prefix.len())
                .step_by(batch_size)
                .map(|a| (a, (a + batch_size).min(prefix.len())))
                .collect()
        } else {
            blocks.to_vec()
        };
        let mut batch = LlamaBatch::new(batch_size, 1);
        for (a, b) in spans {
            batch.clear();
            for (pos, &token) in prefix.iter().enumerate().take(b).skip(a) {
                batch
                    .add(token, pos as i32, &[0], false)
                    .map_err(|e| runtime("Failed to build prefix batch", e))?;
            }
            c.ctx
                .decode(&mut batch)
                .map_err(|e| runtime("Prefix decode failed", e))?;
        }
        c.ctx.synchronize();
        c.prefix = Some((prefix.to_vec(), blocks.to_vec()));
        Ok((ms_since(started), false))
    }

    /// Decode tokens `from..to` of prompt `i` into `seq` in batch-sized
    /// calls, reading the outputs at its read positions after the call that
    /// holds each.
    fn decode_span(
        &mut self,
        batch: &mut LlamaBatch<'_>,
        job: &Job,
        i: usize,
        (from, to): (usize, usize),
        seq: i32,
        out: &mut Vec<Read>,
    ) -> Result<(), DecisionError> {
        let (prompt, reads, readout) = (&job.prompts[i], &job.reads[i], &job.readout);
        let batch_size = self.config.protocol.batch as usize;
        let ctx = &mut self.cached().ctx;
        let mut start = from;
        while start < to {
            let end = (start + batch_size).min(to);
            batch.clear();
            let mut pending = Vec::new();
            for (pos, &token) in prompt.iter().enumerate().take(end).skip(start) {
                let read = reads.binary_search(&pos).is_ok();
                if read {
                    pending.push(batch.n_tokens());
                }
                batch
                    .add(token, pos as i32, &[seq], read)
                    .map_err(|e| runtime("Failed to build question batch", e))?;
            }
            ctx.decode(batch)
                .map_err(|e| runtime("Question decode failed", e))?;
            for index in pending {
                out.push(read_output(ctx, index, readout));
            }
            start = end;
        }
        Ok(())
    }

    /// `SharedPrefix`, causal: the shared part once on sequence 0, then each
    /// prompt's own tokens alone on sequence 1 — a copy of 0's cells,
    /// removed again once its outputs are read. Removing a sequence hands its
    /// cells back and moves llama.cpp's search for free cells to the first
    /// of them, so every prompt lands in the same cells right after the
    /// prefix, with the same attention window: what it computes is the same
    /// whether it is asked alone or among 63 others, in any order. Hybrid
    /// models work the same way: copying a sequence copies its recurrent
    /// state too.
    ///
    /// A shared part that ends on a micro-batch boundary splits nothing a
    /// single call would not: one prompt whose prefix is not kept is then
    /// decoded in one go, with the same result and one call fewer.
    fn isolated(&mut self, job: &Job) -> JobResult {
        let n = job.prompts.len();
        let shared = job.shared;
        let longest_own = job
            .prompts
            .iter()
            .map(|p| p.len() - shared)
            .max()
            .unwrap_or(0);
        // Only one prompt is in the cache at a time.
        self.check_fits(shared + longest_own)?;
        let most_reads = job.reads.iter().map(Vec::len).max().unwrap_or(1);
        let context_ms = self.context(
            job.exact,
            shared + longest_own,
            2,
            u32::try_from(most_reads.max(1)).unwrap_or(u32::MAX),
        )?;

        let prefix = &job.prompts[0][..shared];
        let kept = !prefix.is_empty()
            && self
                .cached()
                .prefix
                .as_ref()
                .is_some_and(|(tokens, _)| tokens == prefix);
        let ubatch = self.config.protocol.ubatch as usize;
        let mut batch = LlamaBatch::new(self.config.protocol.batch as usize, 1);
        let mut outputs: Vec<Vec<Read>> = Vec::with_capacity(n);
        let (prefix_ms, prefix_reused, questions_started, fused);
        if n == 1 && !kept && shared.is_multiple_of(ubatch) {
            fused = true;
            prefix_ms = 0.0;
            prefix_reused = false;
            questions_started = Instant::now();
            clear(&mut self.cached().ctx)?;
            self.cached().prefix = None;
            let mut out = Vec::new();
            let len = job.prompts[0].len();
            self.decode_span(&mut batch, job, 0, (0, len), 0, &mut out)?;
            outputs.push(out);
            // Sequence 0 now holds the question too: nothing to keep.
            clear(&mut self.cached().ctx)?;
        } else {
            fused = false;
            (prefix_ms, prefix_reused) = self.prefix(prefix, &[])?;
            questions_started = Instant::now();
            for i in 0..n {
                if shared > 0 {
                    self.cached()
                        .ctx
                        .copy_kv_cache_seq(0, 1, None, None)
                        .map_err(|e| runtime("Failed to share the prefix", e))?;
                }
                let mut out = Vec::with_capacity(job.reads[i].len());
                let len = job.prompts[i].len();
                self.decode_span(&mut batch, job, i, (shared, len), 1, &mut out)?;
                outputs.push(out);
                drop_sequence(&mut self.cached().ctx, 1)?;
            }
        }
        let questions_ms = ms_since(questions_started);

        let prompt_tokens: usize = job.prompts.iter().map(Vec::len).sum();
        let own: usize = job.prompts.iter().map(|p| p.len() - shared).sum();
        let evaluated_tokens = if fused || !prefix_reused {
            own + shared
        } else {
            own
        };
        let cells = self.cached().cells;
        finish(
            job,
            outputs,
            self.config.threads,
            EvalStats {
                mode: EvalMode::SharedPrefix,
                prompts: n,
                shared_prefix_tokens: shared,
                evaluated_tokens,
                prompt_tokens,
                context_cells: cells,
                context_ms,
                prefix_ms,
                prefix_reused,
                questions_ms,
                readout_ms: 0.0,
            },
        )
    }

    /// `Batched`: the shared part once on sequence 0, then the rest of every
    /// prompt in as few decode calls as the batch size allows, prompt `k` of
    /// a round on sequence `k + 1`. Fewest calls, but a prompt's result moves
    /// with the others in its calls, by the model's own numerical noise.
    fn batched(&mut self, job: &Job) -> JobResult {
        let n = job.prompts.len();
        let shared = job.shared;
        let group = self.config.protocol.group.max(1).min(n);
        let rounds: Vec<std::ops::Range<usize>> = (0..n)
            .step_by(group)
            .map(|a| a..(a + group).min(n))
            .collect();
        let needed = rounds
            .iter()
            .map(|r| {
                shared
                    + job.prompts[r.clone()]
                        .iter()
                        .map(|p| p.len() - shared)
                        .sum::<usize>()
            })
            .max()
            .unwrap_or(shared);
        self.check_fits(needed)?;
        // Outputs one call may ask for: every read of a round, or at least
        // every read of its largest prompt.
        let n_outputs = rounds
            .iter()
            .map(|r| job.reads[r.clone()].iter().map(Vec::len).sum::<usize>())
            .max()
            .unwrap_or(1)
            .clamp(1, 256)
            .max(job.reads.iter().map(Vec::len).max().unwrap_or(1));
        let context_ms = self.context(
            job.exact,
            needed,
            u32::try_from(group + 1).unwrap_or(u32::MAX),
            u32::try_from(n_outputs).unwrap_or(u32::MAX),
        )?;
        let (prefix_ms, prefix_reused) = self.prefix(&job.prompts[0][..shared], &[])?;

        let questions_started = Instant::now();
        let batch_size = self.config.protocol.batch as usize;
        let capacity = self.cached().n_outputs as usize;
        let mut batch = LlamaBatch::new(batch_size, 1);
        let mut outputs: Vec<Vec<Read>> = (0..n).map(|_| Vec::new()).collect();
        for round in &rounds {
            if shared > 0 {
                for k in 1..=round.len() {
                    self.cached()
                        .ctx
                        .copy_kv_cache_seq(0, k as i32, None, None)
                        .map_err(|e| runtime("Failed to share the prefix", e))?;
                }
            }
            batch.clear();
            let mut pending: Vec<(usize, i32)> = Vec::new();
            for (k, i) in round.clone().enumerate() {
                let prompt = &job.prompts[i];
                for (pos, &token) in prompt.iter().enumerate().skip(shared) {
                    let read = job.reads[i].binary_search(&pos).is_ok();
                    if batch.n_tokens() as usize >= batch_size
                        || (read && pending.len() >= capacity)
                    {
                        self.flush(&mut batch, &mut pending, &job.readout, &mut outputs)?;
                    }
                    if read {
                        pending.push((i, batch.n_tokens()));
                    }
                    batch
                        .add(token, pos as i32, &[k as i32 + 1], read)
                        .map_err(|e| runtime("Failed to build question batch", e))?;
                }
            }
            if batch.n_tokens() > 0 {
                self.flush(&mut batch, &mut pending, &job.readout, &mut outputs)?;
            }
            for k in 1..=round.len() {
                drop_sequence(&mut self.cached().ctx, k as u32)?;
            }
        }
        let questions_ms = ms_since(questions_started);

        let prompt_tokens: usize = job.prompts.iter().map(Vec::len).sum();
        let own: usize = job.prompts.iter().map(|p| p.len() - shared).sum();
        let cells = self.cached().cells;
        finish(
            job,
            outputs,
            self.config.threads,
            EvalStats {
                mode: EvalMode::Batched,
                prompts: n,
                shared_prefix_tokens: shared,
                evaluated_tokens: if prefix_reused { own } else { own + shared },
                prompt_tokens,
                context_cells: cells,
                context_ms,
                prefix_ms,
                prefix_reused,
                questions_ms,
                readout_ms: 0.0,
            },
        )
    }

    /// Decode `batch`, wait for it, read the outputs `pending` points at,
    /// and clear both.
    fn flush(
        &mut self,
        batch: &mut LlamaBatch<'_>,
        pending: &mut Vec<(usize, i32)>,
        readout: &Readout,
        outputs: &mut [Vec<Read>],
    ) -> Result<(), DecisionError> {
        let ctx = &mut self.cached().ctx;
        ctx.decode(batch)
            .map_err(|e| runtime("Question decode failed", e))?;
        for &(prompt, index) in pending.iter() {
            outputs[prompt].push(read_output(ctx, index, readout));
        }
        pending.clear();
        batch.clear();
        Ok(())
    }

    /// `Separate`: every prompt from an empty cache, in one run of
    /// batch-sized calls — the baseline.
    fn separate(&mut self, job: &Job) -> JobResult {
        let longest = job.prompts.iter().map(Vec::len).max().unwrap_or(0);
        self.check_fits(longest)?;
        let most_reads = job.reads.iter().map(Vec::len).max().unwrap_or(1);
        let context_ms = self.context(
            job.exact,
            longest,
            1,
            u32::try_from(most_reads.max(1)).unwrap_or(u32::MAX),
        )?;
        self.cached().prefix = None;

        let questions_started = Instant::now();
        let mut batch = LlamaBatch::new(self.config.protocol.batch as usize, 1);
        let mut outputs = Vec::with_capacity(job.prompts.len());
        for i in 0..job.prompts.len() {
            clear(&mut self.cached().ctx)?;
            let mut out = Vec::with_capacity(job.reads[i].len());
            let len = job.prompts[i].len();
            self.decode_span(&mut batch, job, i, (0, len), 0, &mut out)?;
            outputs.push(out);
        }
        clear(&mut self.cached().ctx)?;
        let questions_ms = ms_since(questions_started);

        let prompt_tokens: usize = job.prompts.iter().map(Vec::len).sum();
        let cells = self.cached().cells;
        finish(
            job,
            outputs,
            self.config.threads,
            EvalStats {
                mode: EvalMode::Separate,
                prompts: job.prompts.len(),
                shared_prefix_tokens: 0,
                evaluated_tokens: prompt_tokens,
                prompt_tokens,
                context_cells: cells,
                context_ms,
                prefix_ms: 0.0,
                prefix_reused: false,
                questions_ms,
                readout_ms: 0.0,
            },
        )
    }

    /// Block attention, `SharedPrefix` (and `Batched`): the shared blocks
    /// once on sequence 0, then each prompt's own blocks on sequence 1, one
    /// decode call per block. Nothing after a block can change it, so this
    /// is also exactly what decoding every prompt from scratch computes.
    fn isolated_blocks(&mut self, job: &Job) -> JobResult {
        let n = job.prompts.len();
        let shared = job.shared;
        let longest = job.prompts.iter().map(Vec::len).max().unwrap_or(0);
        self.check_fits(longest)?;
        let context_ms = self.context(job.exact, longest, 2, most_reads_per_block(job))?;
        let shared_blocks: Spans = job.blocks[0]
            .iter()
            .copied()
            .filter(|&(_, b)| b <= shared)
            .collect();
        let (prefix_ms, prefix_reused) = self.prefix(&job.prompts[0][..shared], &shared_blocks)?;

        let questions_started = Instant::now();
        let mut batch = LlamaBatch::new(self.config.protocol.batch as usize, 1);
        let mut outputs = Vec::with_capacity(n);
        for i in 0..n {
            if shared > 0 {
                self.cached()
                    .ctx
                    .copy_kv_cache_seq(0, 1, None, None)
                    .map_err(|e| runtime("Failed to share the prefix", e))?;
            }
            let mut out = Vec::with_capacity(job.reads[i].len());
            for &(a, b) in job.blocks[i].iter().filter(|&&(a, _)| a >= shared) {
                self.decode_span(&mut batch, job, i, (a, b), 1, &mut out)?;
            }
            outputs.push(out);
            drop_sequence(&mut self.cached().ctx, 1)?;
        }
        let questions_ms = ms_since(questions_started);

        let prompt_tokens: usize = job.prompts.iter().map(Vec::len).sum();
        let own: usize = job.prompts.iter().map(|p| p.len() - shared).sum();
        let cells = self.cached().cells;
        finish(
            job,
            outputs,
            self.config.threads,
            EvalStats {
                mode: EvalMode::SharedPrefix,
                prompts: n,
                shared_prefix_tokens: shared,
                evaluated_tokens: if prefix_reused { own } else { own + shared },
                prompt_tokens,
                context_cells: cells,
                context_ms,
                prefix_ms,
                prefix_reused,
                questions_ms,
                readout_ms: 0.0,
            },
        )
    }

    /// Block attention, `Separate`: every prompt's blocks from an empty
    /// cache.
    fn separate_blocks(&mut self, job: &Job) -> JobResult {
        let longest = job.prompts.iter().map(Vec::len).max().unwrap_or(0);
        self.check_fits(longest)?;
        let context_ms = self.context(job.exact, longest, 1, most_reads_per_block(job))?;
        self.cached().prefix = None;

        let questions_started = Instant::now();
        let mut batch = LlamaBatch::new(self.config.protocol.batch as usize, 1);
        let mut outputs = Vec::with_capacity(job.prompts.len());
        for i in 0..job.prompts.len() {
            clear(&mut self.cached().ctx)?;
            let mut out = Vec::with_capacity(job.reads[i].len());
            for &(a, b) in &job.blocks[i] {
                self.decode_span(&mut batch, job, i, (a, b), 0, &mut out)?;
            }
            outputs.push(out);
        }
        clear(&mut self.cached().ctx)?;
        let questions_ms = ms_since(questions_started);

        let prompt_tokens: usize = job.prompts.iter().map(Vec::len).sum();
        let cells = self.cached().cells;
        finish(
            job,
            outputs,
            self.config.threads,
            EvalStats {
                mode: EvalMode::Separate,
                prompts: job.prompts.len(),
                shared_prefix_tokens: 0,
                evaluated_tokens: prompt_tokens,
                prompt_tokens,
                context_cells: cells,
                context_ms,
                prefix_ms: 0.0,
                prefix_reused: false,
                questions_ms,
                readout_ms: 0.0,
            },
        )
    }
}

/// Most read positions any one block holds.
fn most_reads_per_block(job: &Job) -> u32 {
    let most = job
        .reads
        .iter()
        .zip(&job.blocks)
        .flat_map(|(reads, blocks)| {
            blocks
                .iter()
                .map(|&(a, b)| reads.iter().filter(|&&r| r >= a && r < b).count())
        })
        .max()
        .unwrap_or(1);
    u32::try_from(most.max(1)).unwrap_or(u32::MAX)
}

/// Empty the cache — cells only: a free cell's stale contents are masked
/// out of every attention sum, so zeroing them buys nothing. Not
/// `clear_kv_cache_seq(None, ..)`: a recurrent memory refuses to remove
/// "every sequence" and then removes nothing.
fn clear(ctx: &mut LlamaContext<'_>) -> Result<(), DecisionError> {
    ctx.clear_kv_cache_cells();
    Ok(())
}

/// Remove sequence `seq` from the cache, failing loudly if llama.cpp
/// refuses: a question's cells left behind would sit in the next one's way.
fn drop_sequence(ctx: &mut LlamaContext<'_>, seq: u32) -> Result<(), DecisionError> {
    match ctx.clear_kv_cache_seq(Some(seq), None, None) {
        Ok(true) => Ok(()),
        Ok(false) => Err(DecisionError::Runtime(format!(
            "llama.cpp refused to remove sequence {seq} from the decision context"
        ))),
        Err(e) => Err(runtime("Failed to drop a question's cells", e)),
    }
}

/// Copy out what `readout` needs of the output at batch index `index`:
/// the next decode overwrites it.
fn read_output(ctx: &LlamaContext<'_>, index: i32, readout: &Readout) -> Read {
    let logits = ctx.get_logits_ith(index);
    match readout {
        Readout::Classes(_) => Read::Row(logits.to_vec()),
        Readout::Verdict { yes, no } => {
            let at = |t: &LlamaToken| {
                usize::try_from(t.0)
                    .ok()
                    .and_then(|i| logits.get(i))
                    .map_or(f64::NAN, |&l| f64::from(l))
            };
            Read::Score(at(yes) - at(no))
        }
    }
}

/// Turn every prompt's reads into its result and stamp the readout time.
fn finish(job: &Job, outputs: Vec<Vec<Read>>, threads: u32, mut stats: EvalStats) -> JobResult {
    let readout_started = Instant::now();
    let results = match &job.readout {
        Readout::Classes(classes) => {
            let rows: Vec<(usize, &[f32])> = outputs
                .iter()
                .enumerate()
                .map(|(i, reads)| match reads.as_slice() {
                    [Read::Row(row)] => Ok((i, row.as_slice())),
                    _ => Err(DecisionError::Runtime(
                        "a question produced no logits".into(),
                    )),
                })
                .collect::<Result<_, _>>()?;
            let class_refs: Vec<&[Vec<LlamaToken>]> = classes.iter().map(Vec::as_slice).collect();
            let mut results = vec![Vec::new(); outputs.len()];
            for (i, logprobs) in score_rows(&rows, &class_refs, threads as usize)? {
                results[i] = logprobs;
            }
            results
        }
        Readout::Verdict { .. } => outputs
            .into_iter()
            .zip(&job.reads)
            .map(|(reads, wanted)| {
                if reads.len() != wanted.len() {
                    return Err(DecisionError::Runtime(
                        "a question produced fewer scores than it has options".into(),
                    ));
                }
                reads
                    .into_iter()
                    .map(|r| match r {
                        Read::Score(s) if s.is_finite() => Ok(s),
                        _ => Err(DecisionError::Runtime(
                            "the model produced a non-finite score".into(),
                        )),
                    })
                    .collect()
            })
            .collect::<Result<_, _>>()?,
    };
    stats.readout_ms = ms_since(readout_started);
    Ok((results, stats))
}
