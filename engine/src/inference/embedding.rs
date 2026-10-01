//! A second, independent model slot for text embeddings.
//!
//! Deliberately not built on `InferenceEngine`: that type exists to generate
//! text token by token, with a KV cache sized for a whole conversation and a
//! sampler chain behind it. An embedding request is one forward pass per
//! input, discards its own KV cache immediately, and produces a fixed-size
//! vector instead of a token stream. Reusing `InferenceEngine` would mean
//! carrying all of that machinery to switch it back off.
//!
//! Runs alongside the generation model on purpose — see
//! `AppState::ensure_embedding_model` for why keeping both resident (when
//! they fit) beats swapping a 20 GB LLM out for a 500 MB embedder and back on
//! every request.

use std::path::{Path, PathBuf};
use std::pin::pin;
use std::sync::Arc;

use llama_cpp_2::EmbeddingsError;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;

/// Default embedding context. A request to `/api/embed`/`/v1/embeddings` may
/// override it with `options.num_ctx`; a launch-time `--embedding-model`
/// (see `main.rs`) has no per-request body to read one from, so it uses this
/// directly. 2048 comfortably covers a single RAG chunk — the usual unit
/// these are called on — without paying for a window sized to the
/// generation model's much longer conversations.
pub const DEFAULT_EMBEDDING_CTX: u32 = 2048;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

/// A loaded embedding model: a `LlamaModel` bound to the process-wide shared
/// `LlamaBackend` (see `load`), kept alive independently of whatever
/// generation model is loaded in the main slot. Each `embed()` call opens
/// and drops its own `LlamaContext`, sized to that call's longest input (see
/// `embed` for why that size and not the largest one): a short-lived context
/// means an embedding request never competes with a concurrent one for a
/// shared KV cache, and holds its memory only while it runs.
pub struct EmbeddingModel {
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
    /// The GGUF it was loaded from: the same file asked for under another
    /// name is this model (`AppState::ensure_embedding_model`).
    path: PathBuf,
    threads: u32,
    /// Most tokens one input may have. Inputs longer than this are
    /// truncated (see `embed`) rather than rejected — the alternative is a
    /// hard error on the first oversized chunk an ingestion pipeline sends
    /// it, which is a worse failure mode than a documented truncation.
    n_ctx: u32,
}

/// What `EmbeddingModel::embed` produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedded {
    /// One L2-normalized vector per input, in input order.
    pub vectors: Vec<Vec<f32>>,
    /// Tokens the model read, over all inputs and after truncation: what
    /// `usage.prompt_tokens` on `/v1/embeddings` and `prompt_eval_count` on
    /// `/api/embed` report.
    pub prompt_tokens: usize,
}

/// KV cells are allotted in steps of this many: llama.cpp rounds a context
/// up to a multiple of 256 whatever it is asked for, and attends over the
/// used cells rounded up to 256 too. A context sized in these steps is
/// therefore the one llama.cpp would build anyway, and an input sees the
/// same attention window it saw in a context of the full `n_ctx`.
const KV_CELL_STEP: u32 = 256;

/// Batches are sized in steps of this many tokens, so a request of one or two
/// tokens does not build a context around a batch of one.
const BATCH_STEP: u32 = 32;

/// The context one `embed` call needs, as `(n_ctx, n_batch)`: a batch that
/// holds its longest input in a single decode — as the full-size context
/// did, which is what keeps every vector the same — and the KV cells that
/// batch fills. `max` is the most tokens an input may have; inputs have
/// already been cut to it.
fn context_size(longest: usize, max: u32) -> (u32, u32) {
    let max = max.max(1);
    let longest = u32::try_from(longest).unwrap_or(max).clamp(1, max);
    let n_batch = longest.next_multiple_of(BATCH_STEP).min(max);
    (n_batch.next_multiple_of(KV_CELL_STEP), n_batch)
}

impl EmbeddingModel {
    /// Load an embedding model fully onto the GPU (`gpu_layers = -1`) if any
    /// GPU backend is compiled in, CPU otherwise — mirrors
    /// `InferenceEngine::load`'s use of `check_gpu_support`. Embedding models
    /// are small enough (typically 100 MB-1 GB) that a partial CPU/GPU split
    /// is not worth the complexity `--fit` applies to a multi-gigabyte LLM;
    /// full offload or full CPU are the only two shapes this needs.
    ///
    /// `backend` is shared with whatever generation model is loaded in this
    /// process (or loaded later) rather than created fresh here:
    /// `LlamaBackend::init()` marks a process-wide `AtomicBool` and fails
    /// with `BackendAlreadyInitialized` on a second call while the first
    /// instance is still alive — two independently-initialized backends can
    /// never coexist in one process, which used to make the whole point of
    /// this second model slot (staying resident alongside a loaded
    /// generation model) fail with exactly that error on the first request
    /// that tried it. See `main.rs`/`AppState`, which own the one instance
    /// the whole process shares.
    pub fn load(
        path: &Path,
        threads: u32,
        n_ctx: u32,
        backend: Arc<LlamaBackend>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if !path.exists() {
            return Err(format!("Embedding model file not found: {}", path.display()).into());
        }
        // A context of no tokens embeds nothing, and llama.cpp refuses to
        // create one. Refused here, where the value arrives
        // (`options.num_ctx: 0`), rather than as a failure on every request.
        if n_ctx == 0 {
            return Err("an embedding context needs at least one token (num_ctx is 0)".into());
        }

        let gpu_layers = crate::inference::check_gpu_support(-1);
        let model_params = if gpu_layers >= 0 {
            LlamaModelParams::default().with_n_gpu_layers(gpu_layers as u32)
        } else {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        };
        let model_params = pin!(model_params);

        tracing::info!("Loading embedding model: {}", path.display());
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| format!("Failed to load embedding model: {e}"))?;
        tracing::info!(
            "Embedding model loaded — dimension {}",
            model.n_embd()
        );

        Ok(Self {
            backend,
            model,
            path: path.to_path_buf(),
            threads,
            n_ctx,
        })
    }

    pub fn n_embd(&self) -> usize {
        usize::try_from(self.model.n_embd()).unwrap_or(0)
    }

    /// The GGUF this model was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Embed each input text independently, returning one vector per input in
    /// the same order.
    ///
    /// Pooling is deliberately left as `Unspecified`: llama.cpp then reads
    /// the model's own declared pooling type from the GGUF (`hparams`) —
    /// CLS for BGE, mean for E5, and so on — falling back to `None` only for
    /// a model that declares nothing. Overriding it here would silently mask
    /// a real mismatch instead of using the pooling the model was trained
    /// with. When the resolved type genuinely is `None`, this falls back to
    /// mean-pooling the per-token embeddings itself, which is the standard
    /// substitute and better than refusing to answer.
    ///
    /// One text at a time, KV cache cleared between them. Not the fastest
    /// possible shape — batching independent texts into one multi-sequence
    /// decode call would amortize the fixed per-decode cost — but it is the
    /// simple, obviously-correct one, and ingestion throughput here is set
    /// by disk and chunking, not by this loop. Worth revisiting only if
    /// embedding is measured to be the bottleneck.
    ///
    /// The context is sized to the request's longest input (`context_size`),
    /// not to the `n_ctx` the model was loaded with. Building one is not
    /// cheap: llama.cpp allocates and clears the KV cache for every cell, and
    /// reserves compute buffers for the largest batch the context could take
    /// — for a decoder-based embedder that includes a logits row per token,
    /// ~600 KB each for Qwen3's 151k vocabulary, 1.2 GB at 2048 tokens. At
    /// the 2048 default, on a 4-core CPU with Qwen3-Embedding-0.6B, creating
    /// and dropping the context took ~150 ms of the ~240 ms a one-word
    /// request cost, against ~12 ms for a context sized to it; on an RTX 5070
    /// Ti the same requests took 50-110 ms. What is computed does not change:
    /// each input is still decoded whole in one batch, and llama.cpp attends
    /// over the same 256-cell-rounded window in a context of this size as in
    /// the full one. On that CPU the vectors came out bit for bit the same.
    ///
    /// Not a context kept across requests, although that would remove the
    /// rest of the cost: a `LlamaContext` borrows the model, so keeping one
    /// means a worker thread owning both, as `inference::decision` does, and
    /// a context kept at the size of the longest input seen so far holds its
    /// VRAM between requests, which `fit` counts only as a per-request
    /// reserve (`EMBEDDING_COMPUTE_RESERVE_BYTES`).
    pub fn embed(&self, texts: &[String]) -> Result<Embedded, String> {
        if texts.is_empty() {
            return Ok(Embedded {
                vectors: Vec::new(),
                prompt_tokens: 0,
            });
        }

        // Tokenized before the context exists: the longest input decides how
        // large a context this request needs.
        let mut inputs = Vec::with_capacity(texts.len());
        for text in texts {
            let mut tokens = self
                .model
                .str_to_token(text, AddBos::Always)
                .map_err(|e| format!("Tokenization failed: {e}"))?;

            if tokens.len() as u32 > self.n_ctx {
                tracing::warn!(
                    "Embedding input truncated: {} tokens exceeds the {} the embedder was \
                     loaded with",
                    tokens.len(),
                    self.n_ctx
                );
                tokens.truncate(self.n_ctx as usize);
            }
            inputs.push(tokens);
        }
        let prompt_tokens = inputs.iter().map(Vec::len).sum();
        let n_embd = self.n_embd();
        let longest = inputs.iter().map(Vec::len).max().unwrap_or(0);
        if longest == 0 {
            // Nothing to decode, so no context to build.
            return Ok(Embedded {
                vectors: vec![vec![0.0; n_embd]; texts.len()],
                prompt_tokens,
            });
        }

        let (n_ctx, n_batch) = context_size(longest, self.n_ctx);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_batch(n_batch)
            .with_n_ubatch(n_batch)
            .with_n_threads(self.threads as i32)
            .with_n_threads_batch(self.threads as i32)
            .with_n_seq_max(1)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Unspecified);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Failed to create embedding context: {e}"))?;

        let mut vectors = Vec::with_capacity(texts.len());

        for tokens in &inputs {
            if tokens.is_empty() {
                vectors.push(vec![0.0; n_embd]);
                continue;
            }

            ctx.clear_kv_cache();
            let mut batch = LlamaBatch::new(tokens.len(), 1);
            for (i, token) in tokens.iter().enumerate() {
                // logits=true on every token, not just the last: mean/None
                // pooling need every token's embedding, and this matches
                // upstream's own embedding example (`batch_add_seq` in
                // examples/embedding/embedding.cpp) rather than the
                // last-token-only shape generation uses.
                batch
                    .add(*token, i as i32, &[0], true)
                    .map_err(|e| format!("Failed to build embedding batch: {e}"))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| format!("Embedding decode failed: {e}"))?;

            // Ask for the pooled sequence embedding first; the only way the
            // safe wrapper reports "this model's pooling resolved to NONE"
            // is that specific error coming back from the call itself (it
            // does not expose `llama_pooling_type(ctx)` to read up front —
            // see `EmbeddingsError::NonePoolType`), so that is the branch
            // this falls back on rather than predicting it beforehand.
            let mut vector = match ctx.embeddings_seq_ith(0) {
                Ok(v) => v.to_vec(),
                Err(EmbeddingsError::NonePoolType) => mean_pool(&ctx, tokens.len(), n_embd)?,
                Err(e) => return Err(format!("Failed to read embedding: {e}")),
            };

            // L2-normalize so callers can compare with a plain dot product —
            // the convention both the OpenAI and Ollama embedding endpoints
            // follow. Reranker (RANK-pooling) models are out of scope here:
            // normalizing their single relevance scalar would collapse it to
            // +-1 and lose the score entirely, so this endpoint is for
            // embedding models, not rerankers.
            normalize_l2(&mut vector);
            vectors.push(vector);
        }

        Ok(Embedded {
            vectors,
            prompt_tokens,
        })
    }
}

/// Mean-pool per-token embeddings when the model declares no pooling type of
/// its own. `embeddings_ith` returns a reference into the context that
/// borrows `ctx` immutably, so this takes `&LlamaContext` rather than being a
/// method that could conflict with the `&mut ctx` calls around it.
fn mean_pool(
    ctx: &llama_cpp_2::context::LlamaContext,
    n_tokens: usize,
    n_embd: usize,
) -> Result<Vec<f32>, String> {
    let mut sum = vec![0.0f32; n_embd];
    for i in 0..n_tokens {
        let token_embd = ctx
            .embeddings_ith(i as i32)
            .map_err(|e| format!("Failed to read per-token embedding {i}: {e}"))?;
        for (s, v) in sum.iter_mut().zip(token_embd) {
            *s += v;
        }
    }
    let n = n_tokens as f32;
    for s in &mut sum {
        *s /= n;
    }
    Ok(sum)
}

fn normalize_l2(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vector.iter_mut() {
            *v /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A short input gets a small context, not the 2048 default: that
    // context is where a one-word request spent most of its time.
    #[test]
    fn a_short_input_gets_a_small_context() {
        assert_eq!(context_size(2, DEFAULT_EMBEDDING_CTX), (256, 32));
        assert_eq!(context_size(40, DEFAULT_EMBEDDING_CTX), (256, 64));
        assert_eq!(context_size(500, DEFAULT_EMBEDDING_CTX), (512, 512));
    }

    // The batch always takes the longest input in one decode, as the
    // full-size context did — the property that keeps every vector the same.
    #[test]
    fn the_longest_input_always_fits_in_one_batch() {
        for max in [1, 16, 100, 512, 1000, DEFAULT_EMBEDDING_CTX, 8192] {
            for longest in [1, 2, 31, 32, 33, 255, 256, 257, 999, 1000, 2047, 2048, 8192] {
                let longest = longest.min(max as usize);
                let (n_ctx, n_batch) = context_size(longest, max);
                assert!(n_batch as usize >= longest, "{longest} tokens, max {max}");
                assert!(n_batch <= max, "{longest} tokens, max {max}");
                assert!(n_ctx >= n_batch && n_ctx % KV_CELL_STEP == 0);
            }
        }
    }

    // The largest input gets exactly what every request used to get.
    #[test]
    fn the_largest_input_gets_the_full_context() {
        assert_eq!(context_size(2048, 2048), (2048, 2048));
        assert_eq!(context_size(1000, 1000), (1024, 1000));
    }
}
