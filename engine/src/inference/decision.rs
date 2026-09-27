//! Typed decisions read from the model's next-token distribution instead of
//! generated text — the engine side of `POST /v1/systemone`
//! (`api::systemone`).
//!
//! Each question becomes one chat prompt that stops where the model's answer
//! would begin. Nothing is generated: the logits at that position are read
//! once and restricted to a small set of *codes* — `Yes`/`No` for a yes/no
//! question, `A`…`Z` for a choice, `0`…`9` for a score level — each of
//! which must be a single token for this model, which is checked when the
//! model loads (see [`CodeTable`]). What comes back per question is the
//! full-vocabulary log-probability of each code; probabilities, calibration
//! and confidence are derived from those by the pure functions at the end
//! of this module, so the raw numbers are always available next to whatever
//! was derived from them.
//!
//! Questions about the same state share most of their prompt: the system
//! prompt, the template's header and the state come first, the question
//! last. [`EvalMode::SharedPrefix`] decodes that common prefix once on
//! sequence 0, shares its KV cells with one sequence per question
//! (`copy_kv_cache_seq` on a unified cache only tags the existing cells with
//! the new sequence id — no data is copied), then decodes the rest of every
//! question in one batch and reads the logits of each question's last
//! token. For `Q` questions of `q` tokens on a state of `S` that is about
//! `S + Q·q` tokens instead of the `Q·(S + q)` of asking them one at a time,
//! which [`EvalMode::Separate`] still does: it is the reference the shared
//! path is tested for equivalence against and the baseline it is
//! benchmarked against.
//!
//! A separate model slot, like `embedding`, and for the same reason: nothing
//! here needs a sampler, a conversation-sized KV cache or the scheduler, and
//! a small decision model can stay resident next to the chat model.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::Path;
use std::pin::pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

/// Most questions one request may ask. Every question is its own sequence
/// in the shared-prefix batch and gets its own vocabulary-sized logits row
/// (~0.6 MB for a 150k vocabulary), and llama.cpp caps a context at 256
/// sequences; 64 is the top of the range the shared-prefix benchmark
/// measures.
pub const MAX_QUESTIONS: usize = 64;

/// Most options a `choice` question may have: one letter each, `A`…`Z`.
/// The System One API accepts up to 255; going past 26 needs multi-token
/// codes or elimination rounds, which this first version does not do.
pub const MAX_CHOICE_OPTIONS: usize = 26;

/// Fewest options a `choice` question, or levels a `score` question, may
/// have.
pub const MIN_OPTIONS: usize = 2;

/// Most levels a `score` question may have: one digit each, `0`…`9` — the
/// same ceiling as the System One API.
pub const MAX_SCORE_LEVELS: usize = 10;

/// Default ceiling on the KV cells one request's context may use
/// (`--decision-ctx`). The context is sized to each request, so this is not
/// memory held all the time; it is the most one request may ask for, and
/// what the slot reserves VRAM for. 8192 fits a 4k-token state with about
/// fifty short questions, or a 1k-token state with all 64.
pub const DEFAULT_DECISION_CTX: u32 = 8192;

/// The state used to measure a question's content-free prior (Zhao et al.,
/// 2021, "Calibrate Before Use"): the same question asked about no content
/// at all shows which answer the model leans towards before reading
/// anything.
pub const CONTENT_FREE_STATE: &str = "N/A";

/// How `confidence` is computed, reported next to every answer so a client
/// can tell which definition a stored number came from if it ever changes.
pub const CONFIDENCE_METHOD: &str = "normalized_entropy";

/// Upper bound on one decode call. The prefix and the questions' batch are
/// split into chunks of at most this many tokens.
const MAX_BATCH: u32 = 2048;

/// Upper bound on llama.cpp's micro-batch: its own default, and what sizes
/// the compute buffer the slot reserves VRAM for.
const MAX_UBATCH: u32 = 512;

/// Content-free priors kept per loaded model. A prior depends only on the
/// question, so a client asking the same questions about a stream of states
/// pays for it once. Cleared outright when full: the entries are cheap to
/// recompute and a working set larger than this is not the case the cache
/// is for.
const MAX_PRIOR_CACHE: usize = 1024;

/// Floor applied to log-probabilities before they are combined in
/// [`calibrated_probabilities`]. A code the model gives no probability at
/// all would otherwise turn `lp − prior` into `−∞ − (−∞)`.
const LOGPROB_FLOOR: f64 = -1.0e4;

const SYSTEM_PROMPT: &str = "You are a decision function inside a software system. \
Read the state, then answer the question about it. \
Reply with exactly one of the allowed codes and nothing else.";

/// The three question types of the System One API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionKind {
    /// Is the statement true of the state? Answered as P(Yes).
    Noul,
    /// Which one of a set of named options.
    Choice,
    /// Which level of an ordered scale, from low to high.
    Score,
}

impl QuestionKind {
    /// The type's name in the API.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }

    fn code_set(self) -> CodeSet {
        match self {
            Self::Noul => CodeSet::YesNo,
            Self::Choice => CodeSet::Letters,
            Self::Score => CodeSet::Digits,
        }
    }
}

/// One question about the state. The order of `options` and `levels` is the
/// order they are shown to the model and the order of every per-class
/// result: class `i` is option `i` (letter `A` + i), level `i` (digit `i`),
/// or for `Noul` class 0 = Yes, class 1 = No.
#[derive(Debug, Clone, PartialEq)]
pub enum Question {
    Noul {
        instructions: String,
    },
    Choice {
        instructions: String,
        /// `(name, description)`; the description may be empty.
        options: Vec<(String, String)>,
    },
    Score {
        instructions: String,
        /// Level descriptions, lowest first.
        levels: Vec<String>,
    },
}

impl Question {
    pub fn kind(&self) -> QuestionKind {
        match self {
            Self::Noul { .. } => QuestionKind::Noul,
            Self::Choice { .. } => QuestionKind::Choice,
            Self::Score { .. } => QuestionKind::Score,
        }
    }

    pub fn instructions(&self) -> &str {
        match self {
            Self::Noul { instructions }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        }
    }

    /// How many answers the question has: 2 for yes/no, one per option or
    /// level otherwise.
    pub fn n_classes(&self) -> usize {
        match self {
            Self::Noul { .. } => 2,
            Self::Choice { options, .. } => options.len(),
            Self::Score { levels, .. } => levels.len(),
        }
    }

    /// Check the limits every question must respect, whoever built it.
    pub fn validate(&self) -> Result<(), String> {
        if self.instructions().trim().is_empty() {
            return Err("\"instructions\" must not be empty".to_string());
        }
        if self.instructions().contains('\0') {
            return Err("\"instructions\" must not contain NUL characters".to_string());
        }
        match self {
            Self::Noul { .. } => Ok(()),
            Self::Choice { options, .. } => {
                if !(MIN_OPTIONS..=MAX_CHOICE_OPTIONS).contains(&options.len()) {
                    return Err(format!(
                        "a choice question needs {MIN_OPTIONS} to {MAX_CHOICE_OPTIONS} options, got {}",
                        options.len()
                    ));
                }
                for (i, (name, description)) in options.iter().enumerate() {
                    if name.trim().is_empty() {
                        return Err("option names must not be empty".to_string());
                    }
                    if name.contains('\0') || description.contains('\0') {
                        return Err("options must not contain NUL characters".to_string());
                    }
                    if options[..i].iter().any(|(other, _)| other == name) {
                        return Err(format!("duplicate option name \"{name}\""));
                    }
                }
                Ok(())
            }
            Self::Score { levels, .. } => {
                if !(MIN_OPTIONS..=MAX_SCORE_LEVELS).contains(&levels.len()) {
                    return Err(format!(
                        "a score question needs {MIN_OPTIONS} to {MAX_SCORE_LEVELS} levels, got {}",
                        levels.len()
                    ));
                }
                if levels.iter().any(|l| l.trim().is_empty()) {
                    return Err("level descriptions must not be empty".to_string());
                }
                if levels.iter().any(|l| l.contains('\0')) {
                    return Err("levels must not contain NUL characters".to_string());
                }
                Ok(())
            }
        }
    }
}

/// The user turn for one question. The state comes first and the question
/// last on purpose: everything up to the question is shared by every
/// question about the same state, which is what [`EvalMode::SharedPrefix`]
/// decodes only once.
fn user_message(state: &str, question: &Question) -> String {
    use std::fmt::Write as _;

    let mut msg = format!(
        "State:\n{state}\n\nQuestion: {}\n",
        question.instructions().trim()
    );
    match question {
        Question::Noul { .. } => msg.push_str("Answer Yes or No."),
        Question::Choice { options, .. } => {
            msg.push_str("Options:\n");
            for (i, (name, description)) in options.iter().enumerate() {
                let code = CodeSet::Letters.code(i);
                if description.trim().is_empty() {
                    let _ = writeln!(msg, "{code}) {name}");
                } else {
                    let _ = writeln!(msg, "{code}) {name}: {}", description.trim());
                }
            }
            msg.push_str("Answer with the letter of the best option.");
        }
        Question::Score { levels, .. } => {
            msg.push_str("Levels, from lowest to highest:\n");
            for (i, level) in levels.iter().enumerate() {
                let _ = writeln!(msg, "{i}) {}", level.trim());
            }
            msg.push_str("Answer with the number of the level that fits best.");
        }
    }
    msg
}

/// The full prompt for one user turn: the model's own chat template with
/// reasoning switched off, so the answer's first token comes right after the
/// prompt — for Qwen3 that is the pre-closed empty `<think>` block the
/// template renders for `enable_thinking=false`. A model that always
/// reasons regardless (the DeepSeek-R1 family) spends its first token on
/// the opening tag instead, which shows up as a coverage near zero.
///
/// Without an embedded template, a plain-text prompt whose last word asks
/// for the answer.
fn render_prompt(model: &LlamaModel, uses_template: bool, user: &str) -> String {
    if uses_template
        && let Some(rendered) = super::render_jinja_chat_template(
            model,
            &[("system", SYSTEM_PROMPT), ("user", user)],
            false,
        )
    {
        return rendered.prompt;
    }
    format!("{SYSTEM_PROMPT}\n\n{user}\n\nAnswer:")
}

/// The answer codes of one question kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodeSet {
    YesNo,
    Letters,
    Digits,
}

impl CodeSet {
    const ALL: [CodeSet; 3] = [CodeSet::YesNo, CodeSet::Letters, CodeSet::Digits];

    fn index(self) -> usize {
        match self {
            Self::YesNo => 0,
            Self::Letters => 1,
            Self::Digits => 2,
        }
    }

    fn len(self) -> usize {
        match self {
            Self::YesNo => 2,
            Self::Letters => MAX_CHOICE_OPTIONS,
            Self::Digits => MAX_SCORE_LEVELS,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::YesNo => "yes/no",
            Self::Letters => "letter",
            Self::Digits => "digit",
        }
    }

    fn code(self, i: usize) -> String {
        match self {
            Self::YesNo => ["Yes", "No"][i].to_string(),
            Self::Letters => char::from(b'A' + i as u8).to_string(),
            Self::Digits => i.to_string(),
        }
    }

    /// The spellings that count as code `i` when they are the answer's first
    /// token: the code itself and, since most tokenizers fold a leading space
    /// into the word, the code after a space. Yes/No also accepts
    /// lower case; single letters do not, because a lone `a` is an English
    /// word rather than an answer.
    fn forms(self, i: usize) -> Vec<String> {
        let code = self.code(i);
        let mut forms = vec![code.clone(), format!(" {code}")];
        if self == Self::YesNo {
            let lower = code.to_lowercase();
            forms.push(format!(" {lower}"));
            forms.push(lower);
        }
        forms
    }
}

/// For each code of `set`, the token ids that count as that code when the
/// model emits them as the first token of its answer — an empty list for a
/// code this model cannot give in one token.
///
/// `probe` is a fully rendered prompt and `tokenize` must tokenize exactly as
/// real prompts are. A spelling counts only if appending it to the probe
/// adds exactly one token and leaves every probe token as it was. That
/// rejects a spelling that needs two tokens (the probability of its first
/// token would be the probability of a prefix shared with other words) and
/// one that would merge with the end of the prompt (`Answer:` + `A` becoming
/// `Answer` + `:A`), where the model would never see the prompt it is being
/// evaluated on. Checked per model because it depends on the tokenizer and
/// on the template's last characters, not on anything knowable in advance.
fn resolve_codes<F>(set: CodeSet, probe: &str, tokenize: F) -> Result<Vec<Vec<i32>>, String>
where
    F: Fn(&str) -> Result<Vec<i32>, String>,
{
    let base = tokenize(probe)?;
    let mut classes = Vec::with_capacity(set.len());
    for i in 0..set.len() {
        let mut tokens: Vec<i32> = Vec::new();
        for form in set.forms(i) {
            let extended = tokenize(&format!("{probe}{form}"))?;
            if extended.len() == base.len() + 1 && extended[..base.len()] == base[..] {
                let token = extended[base.len()];
                if !tokens.contains(&token) {
                    tokens.push(token);
                }
            }
        }
        classes.push(tokens);
    }

    // A token claimed by two codes would count the same probability twice.
    // No real tokenizer maps two different spellings to one token, so this
    // is a guard rather than a case: drop such a token from both.
    let mut owners: HashMap<i32, usize> = HashMap::new();
    for tokens in &classes {
        for &t in tokens {
            *owners.entry(t).or_default() += 1;
        }
    }
    for tokens in &mut classes {
        tokens.retain(|t| owners[t] == 1);
    }
    Ok(classes)
}

/// Every code's tokens for one loaded model, resolved at load time.
pub struct CodeTable {
    /// Indexed by [`CodeSet::index`], then by class.
    sets: Vec<Vec<Vec<LlamaToken>>>,
}

impl CodeTable {
    fn resolve(model: &LlamaModel, uses_template: bool) -> Result<Self, String> {
        let probe = render_prompt(model, uses_template, "x");
        let tokenize = |text: &str| -> Result<Vec<i32>, String> {
            model
                .str_to_token(text, AddBos::Always)
                .map(|tokens| tokens.into_iter().map(|t| t.0).collect())
                .map_err(|e| format!("Tokenization failed: {e}"))
        };
        let mut sets = Vec::with_capacity(CodeSet::ALL.len());
        for set in CodeSet::ALL {
            let classes = resolve_codes(set, &probe, tokenize)?;
            sets.push(
                classes
                    .into_iter()
                    .map(|tokens| tokens.into_iter().map(LlamaToken).collect())
                    .collect(),
            );
        }
        let table = Self { sets };
        if CodeSet::ALL
            .iter()
            .all(|&set| table.usable_prefix(set) < MIN_OPTIONS.min(set.len()))
        {
            return Err(
                "this model's tokenizer cannot give any decision code (Yes/No, A-Z, 0-9) \
                 as a single token after its prompt"
                    .to_string(),
            );
        }
        Ok(table)
    }

    /// How many leading codes of `set` are usable: a question with more
    /// classes than this cannot be asked.
    fn usable_prefix(&self, set: CodeSet) -> usize {
        self.sets[set.index()]
            .iter()
            .take_while(|tokens| !tokens.is_empty())
            .count()
    }

    /// The token lists of a question's classes, or why this model cannot
    /// answer it.
    fn classes(&self, kind: QuestionKind, n: usize) -> Result<&[Vec<LlamaToken>], String> {
        let set = kind.code_set();
        let classes = &self.sets[set.index()][..n.min(set.len())];
        if let Some(i) = classes.iter().position(Vec::is_empty) {
            return Err(format!(
                "this decision model cannot answer {} questions with {n} classes: the {} code \
                 \"{}\" is not a single token for its tokenizer",
                kind.as_str(),
                set.name(),
                set.code(i)
            ));
        }
        Ok(classes)
    }

    /// One line for the load log: how many codes of each set are usable
    /// and with how many spellings in total.
    fn summary(&self) -> String {
        CodeSet::ALL
            .iter()
            .map(|&set| {
                let classes = &self.sets[set.index()];
                let usable = classes.iter().filter(|t| !t.is_empty()).count();
                let forms: usize = classes.iter().map(Vec::len).sum();
                format!(
                    "{} {usable}/{} ({forms} spellings)",
                    set.name(),
                    classes.len()
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// How the prompts of one request are evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    /// Decode the prefix every prompt shares once, then the rest of every
    /// prompt in one batch. The default.
    SharedPrefix,
    /// Decode every prompt on its own from an empty cache.
    Separate,
}

impl EvalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SharedPrefix => "shared_prefix",
            Self::Separate => "separate",
        }
    }
}

/// What one evaluation cost, reported per request so the shared-prefix
/// saving is measured on real traffic and not only in a benchmark.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalStats {
    pub mode: EvalMode,
    pub prompts: usize,
    /// Tokens every prompt shares, decoded once (`SharedPrefix` only).
    pub shared_prefix_tokens: usize,
    /// Tokens actually decoded.
    pub evaluated_tokens: usize,
    /// Sum of the prompts' lengths: what decoding each on its own costs.
    pub prompt_tokens: usize,
    /// KV cells the context was created with, before llama.cpp's padding.
    pub context_cells: usize,
    pub context_ms: f64,
    /// Decoding the shared prefix (`SharedPrefix` only).
    pub prefix_ms: f64,
    /// Decoding what follows the prefix — every prompt in full for
    /// `Separate`.
    pub questions_ms: f64,
    /// Turning logits rows into per-class log-probabilities. Kept apart
    /// from decode time because it is CPU work over the whole vocabulary.
    pub readout_ms: f64,
}

/// Per-question result of [`DecisionModel::decide`].
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionOutcome {
    /// Full-vocabulary log-probability of each class, in class order.
    pub logprobs: Vec<f64>,
    /// The same, measured on [`CONTENT_FREE_STATE`] — present when a
    /// content-free prior was asked for.
    pub prior_logprobs: Option<Vec<f64>>,
}

/// Result of [`DecisionModel::decide`].
#[derive(Debug, Clone)]
pub struct Decision {
    pub outcomes: Vec<QuestionOutcome>,
    pub stats: EvalStats,
    /// Cost of measuring the content-free priors that were not cached yet;
    /// `None` when none were asked for or all came from the cache.
    pub prior_stats: Option<EvalStats>,
    /// How many priors came from the cache.
    pub priors_cached: usize,
}

/// Options for one [`DecisionModel::decide`] call.
#[derive(Debug, Clone, Copy)]
pub struct DecideOptions {
    pub mode: EvalMode,
    /// Also measure each question's content-free prior.
    pub content_free: bool,
}

/// Why a decision could not be made. The split is the one the API maps to
/// status codes: the first two are the request's to fix (400), the last is
/// ours (500).
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionError {
    /// Not answerable as asked: a limit, an empty field, a code this model's
    /// tokenizer cannot give in one token.
    Invalid(String),
    /// More KV cells than one request may use.
    TooLong { needed: usize, limit: usize },
    /// llama.cpp failed.
    Runtime(String),
}

impl std::fmt::Display for DecisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(msg) | Self::Runtime(msg) => f.write_str(msg),
            Self::TooLong { needed, limit } => write!(
                f,
                "this request needs {needed} tokens of context but the decision model allows \
                 {limit} per request (--decision-ctx): ask fewer questions at once or shorten \
                 the state"
            ),
        }
    }
}

impl std::error::Error for DecisionError {}

fn runtime(what: &str, e: impl std::fmt::Display) -> DecisionError {
    DecisionError::Runtime(format!("{what}: {e}"))
}

/// A loaded decision model: a `LlamaModel` on the process-wide shared
/// backend (see `EmbeddingModel::load` for why it must be shared), the code
/// table resolved against its tokenizer, and a cache of content-free
/// priors.
pub struct DecisionModel {
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
    threads: u32,
    /// Most KV cells one request's context may use.
    max_ctx: u32,
    uses_template: bool,
    codes: CodeTable,
    /// Held for a whole evaluation. Every request creates its own context,
    /// sized to it; two at once would hold two of those in VRAM, and the
    /// slot only reserves room for one (`fit::decision_reserve_bytes`).
    /// One shared-prefix batch already answers all of a request's
    /// questions in parallel, so queueing whole requests costs little.
    eval_lock: Mutex<()>,
    prior_cache: Mutex<HashMap<Vec<LlamaToken>, Vec<f64>>>,
    /// F32 KV cache and no flash attention: takes the cache's own rounding
    /// out of the comparison between shared-prefix and separate decoding,
    /// so on an F32 model the equivalence test checks the logic to ~1e-6
    /// instead of to the model's numerical noise. Only the tests set it; it
    /// costs twice the KV memory.
    exact: bool,
}

impl DecisionModel {
    /// Load a decision model fully onto the GPU if a GPU backend is compiled
    /// in, CPU otherwise — the same two shapes as `EmbeddingModel::load`,
    /// for the same reason: the models this is meant for are a few hundred
    /// megabytes to a few gigabytes. Resolves the code table before
    /// returning, so a model whose tokenizer cannot give any code as one
    /// token is refused at load instead of on its first request.
    pub fn load(
        path: &Path,
        threads: u32,
        max_ctx: u32,
        backend: Arc<LlamaBackend>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if !path.exists() {
            return Err(format!("Decision model file not found: {}", path.display()).into());
        }

        let gpu_layers = super::check_gpu_support(-1);
        let model_params = if gpu_layers >= 0 {
            LlamaModelParams::default().with_n_gpu_layers(gpu_layers as u32)
        } else {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        };
        let model_params = pin!(model_params);

        tracing::info!("Loading decision model: {}", path.display());
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| format!("Failed to load decision model: {e}"))?;

        let uses_template = super::render_jinja_chat_template(
            &model,
            &[("system", SYSTEM_PROMPT), ("user", "x")],
            false,
        )
        .is_some();
        if !uses_template {
            tracing::warn!(
                "Decision model has no usable chat template — using a plain-text prompt, \
                 which instruction-tuned models follow less reliably"
            );
        }
        let codes = CodeTable::resolve(&model, uses_template)?;
        tracing::info!(
            "Decision model loaded — codes: {}; up to {max_ctx} tokens of context per request",
            codes.summary()
        );

        Ok(Self {
            backend,
            model,
            threads: threads.max(1),
            max_ctx,
            uses_template,
            codes,
            eval_lock: Mutex::new(()),
            prior_cache: Mutex::new(HashMap::new()),
            exact: false,
        })
    }

    /// Answer every question about `state`: one full-vocabulary
    /// log-probability per class per question, and optionally the same
    /// measured on [`CONTENT_FREE_STATE`].
    pub fn decide(
        &self,
        state: &str,
        questions: &[Question],
        options: DecideOptions,
    ) -> Result<Decision, DecisionError> {
        if questions.is_empty() {
            return Err(DecisionError::Invalid(
                "at least one question is required".to_string(),
            ));
        }
        if questions.len() > MAX_QUESTIONS {
            return Err(DecisionError::Invalid(format!(
                "at most {MAX_QUESTIONS} questions per request, got {}",
                questions.len()
            )));
        }
        if state.contains('\0') {
            return Err(DecisionError::Invalid(
                "\"state\" must not contain NUL characters".to_string(),
            ));
        }
        let mut classes = Vec::with_capacity(questions.len());
        for question in questions {
            question.validate().map_err(DecisionError::Invalid)?;
            classes.push(
                self.codes
                    .classes(question.kind(), question.n_classes())
                    .map_err(DecisionError::Invalid)?,
            );
        }

        let prompts = self.prompts(state, questions)?;
        let prompt_refs: Vec<&[LlamaToken]> = prompts.iter().map(Vec::as_slice).collect();

        let _running = self
            .eval_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let (logprobs, stats) = self.evaluate(&prompt_refs, &classes, options.mode)?;

        let mut priors: Option<Vec<Vec<f64>>> = None;
        let mut prior_stats = None;
        let mut priors_cached = 0;
        if options.content_free {
            let cf_prompts = self.prompts(CONTENT_FREE_STATE, questions)?;
            let mut found: Vec<Option<Vec<f64>>> = {
                let cache = self
                    .prior_cache
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                cf_prompts.iter().map(|p| cache.get(p).cloned()).collect()
            };
            priors_cached = found.iter().filter(|p| p.is_some()).count();
            let missing: Vec<usize> = (0..found.len()).filter(|&i| found[i].is_none()).collect();
            if !missing.is_empty() {
                let refs: Vec<&[LlamaToken]> =
                    missing.iter().map(|&i| cf_prompts[i].as_slice()).collect();
                let missing_classes: Vec<&[Vec<LlamaToken>]> =
                    missing.iter().map(|&i| classes[i]).collect();
                let (measured, cf_stats) = self.evaluate(&refs, &missing_classes, options.mode)?;
                let mut cache = self
                    .prior_cache
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                if cache.len() + missing.len() > MAX_PRIOR_CACHE {
                    cache.clear();
                }
                for (&i, lp) in missing.iter().zip(measured) {
                    cache.insert(cf_prompts[i].clone(), lp.clone());
                    found[i] = Some(lp);
                }
                prior_stats = Some(cf_stats);
            }
            priors = Some(found.into_iter().map(Option::unwrap_or_default).collect());
        }

        let outcomes = logprobs
            .into_iter()
            .enumerate()
            .map(|(i, logprobs)| QuestionOutcome {
                logprobs,
                prior_logprobs: priors.as_ref().map(|p| p[i].clone()),
            })
            .collect();
        Ok(Decision {
            outcomes,
            stats,
            prior_stats,
            priors_cached,
        })
    }

    fn prompts(
        &self,
        state: &str,
        questions: &[Question],
    ) -> Result<Vec<Vec<LlamaToken>>, DecisionError> {
        questions
            .iter()
            .map(|q| {
                let text = render_prompt(&self.model, self.uses_template, &user_message(state, q));
                self.model
                    .str_to_token(&text, AddBos::Always)
                    .map_err(|e| DecisionError::Invalid(format!("Tokenization failed: {e}")))
            })
            .collect()
    }

    fn evaluate(
        &self,
        prompts: &[&[LlamaToken]],
        classes: &[&[Vec<LlamaToken>]],
        mode: EvalMode,
    ) -> Result<(Vec<Vec<f64>>, EvalStats), DecisionError> {
        let n_ctx_train = self.model.n_ctx_train() as usize;
        if let Some(longest) = prompts.iter().map(|p| p.len()).max()
            && n_ctx_train > 0
            && longest > n_ctx_train
        {
            return Err(DecisionError::Invalid(format!(
                "a prompt of {longest} tokens is longer than the {n_ctx_train} this model was \
                 trained on: shorten the state"
            )));
        }
        match mode {
            EvalMode::SharedPrefix => self.evaluate_shared(prompts, classes),
            EvalMode::Separate => self.evaluate_separate(prompts, classes),
        }
    }

    /// Create a context for one evaluation. `n_seq` sequences, one logits
    /// row each at most per decode call: `n_outputs_max` sized to that
    /// rather than left at its default of `n_batch`, which would reserve a
    /// vocabulary-sized row per batch slot in the compute buffer. The cache
    /// is unified so the sequences can share the prefix's cells; without it
    /// llama.cpp gives each sequence its own `n_ctx / n_seq` slice and
    /// copying the prefix would copy the data.
    fn context(
        &self,
        cells: usize,
        n_batch: u32,
        n_seq: u32,
    ) -> Result<LlamaContext<'_>, DecisionError> {
        let n_ctx = u32::try_from(cells).unwrap_or(u32::MAX).max(1);
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_batch)
            .with_n_ubatch(n_batch.min(MAX_UBATCH))
            .with_n_seq_max(n_seq)
            .with_n_outputs_max(n_seq)
            .with_kv_unified(true)
            .with_n_threads(self.threads as i32)
            .with_n_threads_batch(self.threads as i32);
        let params = if self.exact {
            params
                .with_type_k(KvCacheType::F32)
                .with_type_v(KvCacheType::F32)
                .with_flash_attention_policy(0)
        } else {
            params
        };
        self.model
            .new_context(&self.backend, params)
            .map_err(|e| DecisionError::Runtime(format!("Failed to create decision context: {e}")))
    }

    fn evaluate_shared(
        &self,
        prompts: &[&[LlamaToken]],
        classes: &[&[Vec<LlamaToken>]],
    ) -> Result<(Vec<Vec<f64>>, EvalStats), DecisionError> {
        let started = Instant::now();
        let n = prompts.len();
        let prompt_tokens: usize = prompts.iter().map(|p| p.len()).sum();
        let prefix_len = shared_prefix_len(prompts);
        let suffix_total = prompt_tokens - n * prefix_len;
        let cells = prefix_len + suffix_total;
        if cells > self.max_ctx as usize {
            return Err(DecisionError::TooLong {
                needed: cells,
                limit: self.max_ctx as usize,
            });
        }

        let n_batch = u32::try_from(prefix_len.max(suffix_total))
            .unwrap_or(MAX_BATCH)
            .clamp(1, MAX_BATCH);
        let mut ctx = self.context(cells, n_batch, n as u32)?;
        let context_ms = ms_since(started);

        // 1. The shared prefix, once, on sequence 0. No logits: every
        //    prompt has at least one token after it (`shared_prefix_len`).
        let prefix_started = Instant::now();
        let mut batch = LlamaBatch::new(n_batch as usize, 1);
        for (chunk_index, chunk) in prompts[0][..prefix_len]
            .chunks(n_batch as usize)
            .enumerate()
        {
            batch.clear();
            let base = chunk_index * n_batch as usize;
            for (j, &token) in chunk.iter().enumerate() {
                batch
                    .add(token, (base + j) as i32, &[0], false)
                    .map_err(|e| runtime("Failed to build prefix batch", e))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| runtime("Prefix decode failed", e))?;
        }
        // 2. Every other sequence starts from the same cells.
        if prefix_len > 0 {
            for seq in 1..n {
                ctx.copy_kv_cache_seq(0, seq as i32, None, None)
                    .map_err(|e| runtime("Failed to share the prefix", e))?;
            }
        }
        let prefix_ms = ms_since(prefix_started);

        // 3. The rest of every prompt, all in the same batches, with logits
        //    on each prompt's last token only.
        let questions_started = Instant::now();
        let mut readout_ms = 0.0;
        let mut results: Vec<Option<Vec<f64>>> = vec![None; n];
        let mut pending: Vec<(usize, i32)> = Vec::new();
        batch.clear();
        for (i, prompt) in prompts.iter().enumerate() {
            let suffix = &prompt[prefix_len..];
            for (j, &token) in suffix.iter().enumerate() {
                if batch.n_tokens() as u32 >= n_batch {
                    readout_ms +=
                        self.flush(&mut ctx, &mut batch, &mut pending, classes, &mut results)?;
                }
                let last = j + 1 == suffix.len();
                if last {
                    pending.push((i, batch.n_tokens()));
                }
                batch
                    .add(token, (prefix_len + j) as i32, &[i as i32], last)
                    .map_err(|e| runtime("Failed to build question batch", e))?;
            }
        }
        if batch.n_tokens() > 0 {
            readout_ms += self.flush(&mut ctx, &mut batch, &mut pending, classes, &mut results)?;
        }
        let questions_ms = ms_since(questions_started) - readout_ms;

        let logprobs = results
            .into_iter()
            .map(|r| {
                r.ok_or_else(|| DecisionError::Runtime("a question produced no logits".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((
            logprobs,
            EvalStats {
                mode: EvalMode::SharedPrefix,
                prompts: n,
                shared_prefix_tokens: prefix_len,
                evaluated_tokens: cells,
                prompt_tokens,
                context_cells: cells,
                context_ms,
                prefix_ms,
                questions_ms,
                readout_ms,
            },
        ))
    }

    /// Decode `batch`, read the rows `pending` points at, and clear both.
    /// Returns the milliseconds spent reading.
    fn flush(
        &self,
        ctx: &mut LlamaContext<'_>,
        batch: &mut LlamaBatch<'_>,
        pending: &mut Vec<(usize, i32)>,
        classes: &[&[Vec<LlamaToken>]],
        results: &mut [Option<Vec<f64>>],
    ) -> Result<f64, DecisionError> {
        ctx.decode(batch)
            .map_err(|e| runtime("Question decode failed", e))?;
        let started = Instant::now();
        for (prompt, logprobs) in read_rows(ctx, pending, classes, self.threads as usize)? {
            results[prompt] = Some(logprobs);
        }
        pending.clear();
        batch.clear();
        Ok(ms_since(started))
    }

    fn evaluate_separate(
        &self,
        prompts: &[&[LlamaToken]],
        classes: &[&[Vec<LlamaToken>]],
    ) -> Result<(Vec<Vec<f64>>, EvalStats), DecisionError> {
        let started = Instant::now();
        let prompt_tokens: usize = prompts.iter().map(|p| p.len()).sum();
        let longest = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        if longest > self.max_ctx as usize {
            return Err(DecisionError::TooLong {
                needed: longest,
                limit: self.max_ctx as usize,
            });
        }
        let n_batch = u32::try_from(longest)
            .unwrap_or(MAX_BATCH)
            .clamp(1, MAX_BATCH);
        let mut ctx = self.context(longest, n_batch, 1)?;
        let context_ms = ms_since(started);

        let questions_started = Instant::now();
        let mut readout_ms = 0.0;
        let mut results: Vec<Option<Vec<f64>>> = vec![None; prompts.len()];
        let mut pending = Vec::with_capacity(1);
        let mut batch = LlamaBatch::new(n_batch as usize, 1);
        for (i, prompt) in prompts.iter().enumerate() {
            ctx.clear_kv_cache();
            let n_chunks = prompt.len().div_ceil(n_batch as usize);
            for (chunk_index, chunk) in prompt.chunks(n_batch as usize).enumerate() {
                batch.clear();
                let base = chunk_index * n_batch as usize;
                for (j, &token) in chunk.iter().enumerate() {
                    let last = chunk_index + 1 == n_chunks && j + 1 == chunk.len();
                    if last {
                        pending.push((i, batch.n_tokens()));
                    }
                    batch
                        .add(token, (base + j) as i32, &[0], last)
                        .map_err(|e| runtime("Failed to build prompt batch", e))?;
                }
                if chunk_index + 1 < n_chunks {
                    ctx.decode(&mut batch)
                        .map_err(|e| runtime("Prompt decode failed", e))?;
                }
            }
            readout_ms += self.flush(&mut ctx, &mut batch, &mut pending, classes, &mut results)?;
        }
        let questions_ms = ms_since(questions_started) - readout_ms;

        let logprobs = results
            .into_iter()
            .map(|r| {
                r.ok_or_else(|| DecisionError::Runtime("a question produced no logits".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((
            logprobs,
            EvalStats {
                mode: EvalMode::Separate,
                prompts: prompts.len(),
                shared_prefix_tokens: 0,
                evaluated_tokens: prompt_tokens,
                prompt_tokens,
                context_cells: longest,
                context_ms,
                prefix_ms: 0.0,
                questions_ms,
                readout_ms,
            },
        ))
    }
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Tokens every prompt starts with, capped one short of the shortest prompt
/// so every prompt keeps at least one token of its own — the one whose
/// logits are read.
fn shared_prefix_len(prompts: &[&[LlamaToken]]) -> usize {
    let Some((first, rest)) = prompts.split_first() else {
        return 0;
    };
    let mut len = first.len();
    for prompt in rest {
        len = first[..len]
            .iter()
            .zip(prompt.iter())
            .take_while(|(a, b)| a == b)
            .count();
    }
    let shortest = prompts.iter().map(|p| p.len()).min().unwrap_or(0);
    len.min(shortest.saturating_sub(1))
}

/// Per-class log-probabilities of the rows `pending` points at, as
/// `(prompt, logprobs)`. The rows are read out of the context first and
/// then scored on up to `workers` threads: each row is a pass over the whole
/// vocabulary (~0.5-0.9 ms for 150k entries on one core, measured), which
/// for 64 questions would otherwise add tens of milliseconds to a batch the
/// GPU finished long before.
fn read_rows(
    ctx: &LlamaContext<'_>,
    pending: &[(usize, i32)],
    classes: &[&[Vec<LlamaToken>]],
    workers: usize,
) -> Result<Vec<(usize, Vec<f64>)>, DecisionError> {
    let rows: Vec<(usize, &[f32])> = pending
        .iter()
        .map(|&(prompt, index)| (prompt, ctx.get_logits_ith(index)))
        .collect();
    let score = |part: &[(usize, &[f32])]| -> Result<Vec<(usize, Vec<f64>)>, DecisionError> {
        part.iter()
            .map(|&(prompt, logits)| class_logprobs(logits, classes[prompt]).map(|lp| (prompt, lp)))
            .collect()
    };
    let workers = workers.clamp(1, rows.len().max(1));
    if workers == 1 {
        return score(&rows);
    }
    let per_worker = rows.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(per_worker)
            .map(|part| scope.spawn(move || score(part)))
            .collect();
        let mut out = Vec::with_capacity(rows.len());
        for handle in handles {
            let part = handle
                .join()
                .map_err(|_| DecisionError::Runtime("logits readout thread panicked".into()))??;
            out.extend(part);
        }
        Ok(out)
    })
}

/// Full-vocabulary log-probability of each class: the log of the summed
/// probabilities of its spellings.
fn class_logprobs(logits: &[f32], classes: &[Vec<LlamaToken>]) -> Result<Vec<f64>, DecisionError> {
    let norm = vocab_log_normalizer(logits);
    if !norm.is_finite() {
        return Err(DecisionError::Runtime(
            "the model produced non-finite logits".to_string(),
        ));
    }
    Ok(classes
        .iter()
        .map(|tokens| {
            let lps: Vec<f64> = tokens
                .iter()
                .map(|t| {
                    usize::try_from(t.0)
                        .ok()
                        .and_then(|i| logits.get(i))
                        .map_or(f64::NEG_INFINITY, |&l| f64::from(l) - norm)
                })
                .collect();
            log_sum_exp(&lps)
        })
        .collect())
}

/// `log Σ exp(logit)` over a whole logits row. Exponentials in `f32`,
/// summed in chunks of 1024 and the chunks accumulated in `f64`: about 40%
/// faster than all-`f64` on a 150k vocabulary, and a relative error far
/// below anything the probabilities are reported to. `NaN` if any logit is.
fn vocab_log_normalizer(logits: &[f32]) -> f64 {
    if logits.iter().any(|l| l.is_nan()) {
        return f64::NAN;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f64::from(max);
    }
    let total: f64 = logits
        .chunks(1024)
        .map(|chunk| f64::from(chunk.iter().map(|&l| (l - max).exp()).sum::<f32>()))
        .sum();
    f64::from(max) + total.ln()
}

/// `log Σ exp(v)`, stable for any magnitudes; `−∞` for no values or all
/// `−∞`.
pub fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
}

/// Total probability the model put on the valid codes, out of its whole
/// vocabulary. Near 1: it answered in the format asked for. Low: most of its
/// probability went to something else (a thinking tag, a sentence, another
/// language), and the renormalized probabilities describe a minority of
/// what it would actually have said.
pub fn coverage(logprobs: &[f64]) -> f64 {
    logprobs
        .iter()
        .map(|lp| lp.exp())
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// Class probabilities from class log-probabilities: renormalized over the
/// classes, after dividing out `prior` when given (the content-free
/// correction, `q ∝ p / p_cf` — the argmax of Zhao et al.'s diagonal
/// calibration, renormalized so it stays a distribution) and scaling by
/// `1 / temperature` (temperature scaling: `T > 1` flattens an
/// overconfident model, `T < 1` sharpens). `prior = None` and `T = 1` give
/// the raw probabilities. A `temperature` that is not a positive finite
/// number is treated as 1; the API refuses one before it gets here.
pub fn calibrated_probabilities(
    logprobs: &[f64],
    prior: Option<&[f64]>,
    temperature: f64,
) -> Vec<f64> {
    let t = if temperature.is_finite() && temperature > 0.0 {
        temperature
    } else {
        1.0
    };
    let adjusted: Vec<f64> = logprobs
        .iter()
        .enumerate()
        .map(|(i, &lp)| {
            let prior = prior.and_then(|p| p.get(i)).copied().unwrap_or(0.0);
            (lp.max(LOGPROB_FLOOR) - prior.max(LOGPROB_FLOOR)) / t
        })
        .collect();
    let norm = log_sum_exp(&adjusted);
    adjusted.iter().map(|a| (a - norm).exp()).collect()
}

/// `1 − H(p) / ln K`: 1 when all probability is on one class, 0 when it is
/// spread evenly over all `K`. One of several reasonable definitions — the
/// System One API only says confidence is "computed from how probabilities
/// is spread" — which is why its name travels with it
/// ([`CONFIDENCE_METHOD`]).
pub fn normalized_entropy_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let entropy: f64 = probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    (1.0 - entropy / (probabilities.len() as f64).ln()).clamp(0.0, 1.0)
}

/// `Σ level × p(level)`: a position on the scale, between levels when the
/// model is split between them — the System One definition of `score`.
pub fn expected_level(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .map(|(level, p)| level as f64 * p)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn log_sum_exp_matches_the_direct_formula_and_survives_large_values() {
        let v = [0.5, -1.0, 2.0];
        let direct = v.iter().map(|x: &f64| x.exp()).sum::<f64>().ln();
        assert!(close(log_sum_exp(&v), direct));
        // exp(1000) overflows; the shifted form does not.
        assert!(close(log_sum_exp(&[1000.0, 1000.0]), 1000.0 + 2f64.ln()));
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(&[f64::NEG_INFINITY]), f64::NEG_INFINITY);
    }

    #[test]
    fn vocab_normalizer_agrees_with_f64_on_a_realistic_row() {
        // A 150k row with a spread like real logits: the chunked f32 sum
        // must agree with a straightforward f64 one to well under the
        // precision probabilities are reported with.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let row: Vec<f32> = (0..151_936)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % 10_000) as f32 / 10_000.0 * 30.0 - 10.0
            })
            .collect();
        let exact = log_sum_exp(&row.iter().map(|&l| f64::from(l)).collect::<Vec<_>>());
        assert!((vocab_log_normalizer(&row) - exact).abs() < 1e-5);
        assert!(vocab_log_normalizer(&[1.0, f32::NAN]).is_nan());
    }

    #[test]
    fn class_logprobs_sum_the_spellings_of_each_class() {
        // Vocabulary of 4 with logits ln(1), ln(2), ln(3), ln(4): token
        // probabilities 0.1, 0.2, 0.3, 0.4.
        let logits: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|v| v.ln()).collect();
        let classes = vec![vec![LlamaToken(0), LlamaToken(2)], vec![LlamaToken(3)]];
        let lp = class_logprobs(&logits, &classes).unwrap();
        assert!((lp[0].exp() - 0.4).abs() < 1e-6);
        assert!((lp[1].exp() - 0.4).abs() < 1e-6);
        assert!((coverage(&lp) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn class_logprobs_refuse_non_finite_logits() {
        let classes = vec![vec![LlamaToken(0)], vec![LlamaToken(1)]];
        assert!(class_logprobs(&[f32::NAN, 1.0], &classes).is_err());
    }

    #[test]
    fn uncalibrated_probabilities_are_the_renormalized_class_probabilities() {
        // Classes holding 0.6 and 0.2 of the vocabulary → 0.75 / 0.25.
        let lp = [0.6f64.ln(), 0.2f64.ln()];
        let p = calibrated_probabilities(&lp, None, 1.0);
        assert!(close(p[0], 0.75) && close(p[1], 0.25));
    }

    #[test]
    fn a_content_free_prior_is_divided_out() {
        // The model says 0.75/0.25 about the state but already said
        // 0.75/0.25 about nothing at all: once its lean is removed, the
        // state carries no evidence either way.
        let lp = [0.6f64.ln(), 0.2f64.ln()];
        let prior = [0.3f64.ln(), 0.1f64.ln()];
        let p = calibrated_probabilities(&lp, Some(&prior), 1.0);
        assert!(close(p[0], 0.5) && close(p[1], 0.5));

        // A prior leaning the other way reinforces the answer instead.
        let prior = [0.25f64.ln(), 0.75f64.ln()];
        let p = calibrated_probabilities(&lp, Some(&prior), 1.0);
        assert!(close(p[0], 0.9));
    }

    #[test]
    fn temperature_flattens_or_sharpens_without_changing_the_ranking() {
        let lp = [0.7f64.ln(), 0.2f64.ln(), 0.1f64.ln()];
        let raw = calibrated_probabilities(&lp, None, 1.0);
        let flat = calibrated_probabilities(&lp, None, 2.0);
        let sharp = calibrated_probabilities(&lp, None, 0.5);
        assert!(flat[0] < raw[0] && raw[0] < sharp[0]);
        for p in [&raw, &flat, &sharp] {
            assert!(close(p.iter().sum(), 1.0));
            assert!(p[0] > p[1] && p[1] > p[2]);
        }
        // T = 2 is the square root of the odds.
        assert!(close(flat[0] / flat[1], (0.7f64 / 0.2).sqrt()));
    }

    #[test]
    fn an_invalid_temperature_is_treated_as_one() {
        let lp = [0.7f64.ln(), 0.3f64.ln()];
        let raw = calibrated_probabilities(&lp, None, 1.0);
        for t in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(calibrated_probabilities(&lp, None, t), raw);
        }
    }

    #[test]
    fn a_class_with_no_probability_at_all_stays_finite() {
        let lp = [f64::NEG_INFINITY, 0.0];
        let prior = [f64::NEG_INFINITY, 0.0];
        let p = calibrated_probabilities(&lp, Some(&prior), 1.0);
        assert!(p.iter().all(|v| v.is_finite()));
        assert!(close(p.iter().sum(), 1.0));
    }

    #[test]
    fn confidence_is_one_for_certainty_and_zero_for_an_even_split() {
        assert!(close(normalized_entropy_confidence(&[1.0, 0.0, 0.0]), 1.0));
        assert!(close(normalized_entropy_confidence(&[0.25; 4]), 0.0));
        let c = normalized_entropy_confidence(&[0.9, 0.1]);
        assert!(c > 0.5 && c < 1.0);
    }

    #[test]
    fn expected_level_matches_the_system_one_example() {
        // "0 x 0.0 + 1 x 0.57 + 2 x 0.43 = 1.43"
        assert!(close(expected_level(&[0.0, 0.57, 0.43]), 1.43));
    }

    #[test]
    fn codes_are_shown_as_letters_digits_and_yes_no() {
        let code = |kind: QuestionKind, i| kind.code_set().code(i);
        assert_eq!(code(QuestionKind::Choice, 0), "A");
        assert_eq!(code(QuestionKind::Choice, 25), "Z");
        assert_eq!(code(QuestionKind::Score, 9), "9");
        assert_eq!(code(QuestionKind::Noul, 0), "Yes");
        assert_eq!(code(QuestionKind::Noul, 1), "No");
        assert_eq!(CodeSet::YesNo.forms(0), vec!["Yes", " Yes", " yes", "yes"]);
        assert_eq!(CodeSet::Letters.forms(1), vec!["B", " B"]);
    }

    /// A toy tokenizer: greedy longest match against a small vocabulary,
    /// falling back to one token per character (ids from 1000) like the
    /// byte fallback of a real one. `"Answer:A"` is in the vocabulary to
    /// stand for the merges a real BPE makes when a code attaches to the
    /// prompt's last word.
    fn toy_tokenize(text: &str) -> Result<Vec<i32>, String> {
        const VOCAB: &[&str] = &[
            "Answer:", "\n", "A", " A", "B", " B", "Yes", " Yes", "No", " No", "yes", "no",
            "Answer:A",
        ];
        let mut out = Vec::new();
        let mut rest = text;
        while let Some(c) = rest.chars().next() {
            let (id, len) = VOCAB
                .iter()
                .enumerate()
                .filter(|(_, v)| rest.starts_with(**v))
                .map(|(i, v)| (i as i32, v.len()))
                .max_by_key(|&(_, len)| len)
                .unwrap_or((1000 + c as i32, c.len_utf8()));
            out.push(id);
            rest = &rest[len..];
        }
        Ok(out)
    }

    #[test]
    fn a_code_that_merges_with_the_prompt_does_not_count() {
        // "Answer:" + "A" tokenizes as the single token "Answer:A", which
        // rewrites the probe's last token — only " A" is a clean next token.
        let classes = resolve_codes(CodeSet::Letters, "Answer:", toy_tokenize).unwrap();
        assert_eq!(classes[0], vec![3], "only the spaced form of A");
        assert_eq!(classes[1], vec![4, 5], "both forms of B");
        // " C" falls back to two character tokens; "C" alone is one.
        assert_eq!(classes[2], vec![1000 + 'C' as i32]);
    }

    #[test]
    fn a_spelling_that_needs_two_tokens_does_not_count() {
        // " yes" is a space token followed by "yes": its first token would
        // be the probability of a bare space, not of the answer.
        let classes = resolve_codes(CodeSet::YesNo, "\n", toy_tokenize).unwrap();
        assert_eq!(classes[0], vec![6, 7, 10], "Yes, ' Yes', yes");
        assert_eq!(classes[1], vec![8, 9, 11], "No, ' No', no");
    }

    #[test]
    fn a_token_claimed_by_two_codes_is_dropped_from_both() {
        // A tokenizer that maps "A" and "B" to the same id.
        let tokenize = |text: &str| -> Result<Vec<i32>, String> {
            Ok(text
                .chars()
                .map(|c| match c {
                    'A' | 'B' => 1,
                    ' ' => 0,
                    other => other as i32,
                })
                .collect())
        };
        let classes = resolve_codes(CodeSet::Letters, "\n", tokenize).unwrap();
        assert!(classes[0].is_empty() && classes[1].is_empty());
        assert!(!classes[2].is_empty());
    }

    #[test]
    fn the_code_table_names_the_first_code_a_question_cannot_use() {
        let token = |t| vec![LlamaToken(t)];
        let mut letters: Vec<Vec<LlamaToken>> = (0..26).map(token).collect();
        letters[3] = Vec::new(); // "D" unusable
        let table = CodeTable {
            sets: vec![
                vec![token(100), token(101)],
                letters,
                (200..210).map(token).collect(),
            ],
        };
        assert!(table.classes(QuestionKind::Choice, 3).is_ok());
        let err = table.classes(QuestionKind::Choice, 4).unwrap_err();
        assert!(err.contains("\"D\""), "{err}");
        assert_eq!(table.usable_prefix(CodeSet::Letters), 3);
        assert_eq!(table.classes(QuestionKind::Noul, 2).unwrap().len(), 2);
    }

    #[test]
    fn the_shared_prefix_leaves_every_prompt_a_token_of_its_own() {
        let t = |v: &[i32]| v.iter().map(|&x| LlamaToken(x)).collect::<Vec<_>>();
        let (a, b, c) = (t(&[1, 2, 3, 4]), t(&[1, 2, 3, 5, 6]), t(&[1, 2, 9]));
        assert_eq!(shared_prefix_len(&[&a, &b]), 3);
        assert_eq!(shared_prefix_len(&[&a, &b, &c]), 2);
        // Identical prompts: everything but the last token is shared.
        assert_eq!(shared_prefix_len(&[&a, &a]), 3);
        // One prompt: same, so a single question takes the same path.
        assert_eq!(shared_prefix_len(&[&b]), 4);
        // One prompt a prefix of another: the shorter keeps its last token.
        let d = t(&[1, 2]);
        assert_eq!(shared_prefix_len(&[&a, &d]), 1);
        assert_eq!(shared_prefix_len(&[]), 0);
    }

    #[test]
    fn the_state_comes_before_the_question_in_every_prompt() {
        let q1 = Question::Noul {
            instructions: "Is it urgent?".into(),
        };
        let q2 = Question::Choice {
            instructions: "Which team?".into(),
            options: vec![
                ("billing".into(), "Payments".into()),
                ("other".into(), String::new()),
            ],
        };
        let q3 = Question::Score {
            instructions: "How severe?".into(),
            levels: vec!["Cosmetic".into(), "Blocking".into()],
        };
        let state = "Payouts failing for 3 days.";
        let prefix = format!("State:\n{state}\n\nQuestion: ");
        for q in [&q1, &q2, &q3] {
            assert!(user_message(state, q).starts_with(&prefix));
        }
        let choice = user_message(state, &q2);
        assert!(
            choice.contains("A) billing: Payments\nB) other\n"),
            "{choice}"
        );
        let score = user_message(state, &q3);
        assert!(score.contains("0) Cosmetic\n1) Blocking\n"), "{score}");
        assert!(user_message(state, &q1).ends_with("Answer Yes or No."));
    }

    #[test]
    fn question_limits_are_enforced() {
        let options = |n: usize| -> Vec<(String, String)> {
            (0..n).map(|i| (format!("o{i}"), String::new())).collect()
        };
        let choice = |n| Question::Choice {
            instructions: "Which?".into(),
            options: options(n),
        };
        assert!(choice(1).validate().is_err());
        assert!(choice(2).validate().is_ok());
        assert!(choice(26).validate().is_ok());
        assert!(choice(27).validate().is_err());

        let score = |n: usize| Question::Score {
            instructions: "How much?".into(),
            levels: (0..n).map(|i| format!("level {i}")).collect(),
        };
        assert!(score(1).validate().is_err());
        assert!(score(10).validate().is_ok());
        assert!(score(11).validate().is_err());

        let dup = Question::Choice {
            instructions: "Which?".into(),
            options: vec![("a".into(), String::new()), ("a".into(), String::new())],
        };
        assert!(dup.validate().unwrap_err().contains("duplicate"));
        let empty = Question::Noul {
            instructions: "  ".into(),
        };
        assert!(empty.validate().is_err());
        let nul = Question::Noul {
            instructions: "a\0b".into(),
        };
        assert!(nul.validate().is_err());
    }

    #[test]
    fn too_long_names_the_flag_to_raise() {
        let msg = DecisionError::TooLong {
            needed: 9000,
            limit: 8192,
        }
        .to_string();
        assert!(msg.contains("9000") && msg.contains("8192") && msg.contains("--decision-ctx"));
    }

    /// The shared-prefix batch must give every question the answer it gets
    /// on its own. A mistake in positions, sequence ids or cell sharing
    /// does not fail loudly — a question attending to another's tokens
    /// still yields a plausible distribution — so this compares the two
    /// paths on a real model, with an F32 KV cache and no flash attention
    /// (`DecisionModel::exact`) to take the cache's own rounding out.
    ///
    /// What is left is the model's arithmetic. On F32 weights the two paths
    /// agree to ~2e-6 (stories260K, measured), which is what the default
    /// tolerance is for. On quantized weights they do not, and cannot: a
    /// question whose cells sit at other indices of the cache sums its
    /// attention in another order, and the 8-bit activation quantization
    /// of a Q8_0 model turns that last-digit difference into a different
    /// rounding one layer later — measured on Qwen3-0.6B: 1e-2 (F16) and up
    /// to 0.67 nats (Q8_0) on the same questions. Run it on an F32 model
    /// (the 1.2 MB `stories260K.gguf` llama.cpp's CI uses is enough); a
    /// quantized one needs `EULLM_DECISION_TEST_TOLERANCE` and proves less.
    ///
    /// ```text
    /// EULLM_DECISION_TEST_MODEL=/path/to/stories260K.gguf \
    ///     cargo test --bin eullm shared_prefix_matches -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a GGUF model in EULLM_DECISION_TEST_MODEL"]
    fn shared_prefix_matches_separate_evaluation() {
        let path = std::env::var("EULLM_DECISION_TEST_MODEL")
            .expect("set EULLM_DECISION_TEST_MODEL to a GGUF file");
        let tolerance: f64 = std::env::var("EULLM_DECISION_TEST_TOLERANCE")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(1e-3);
        let backend = crate::inference::init_shared_backend().expect("backend");
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as u32);
        let mut model =
            DecisionModel::load(Path::new(&path), threads, DEFAULT_DECISION_CTX, backend)
                .expect("load the model");
        model.exact = true;

        let state = "Tom had a red ball. He lost it in the park on Monday. Lily found it \
                     under a tree and gave it back to him, and they played until dinner.";
        let all = vec![
            Question::Noul {
                instructions: "Did Lily give the ball back?".into(),
            },
            Question::Choice {
                instructions: "Who found the ball?".into(),
                options: vec![
                    ("tom".into(), "Tom".into()),
                    ("lily".into(), "Lily".into()),
                    ("dog".into(), "A dog".into()),
                ],
            },
            Question::Score {
                instructions: "How happy is the ending?".into(),
                levels: vec!["Sad".into(), "Neutral".into(), "Happy".into()],
            },
            Question::Choice {
                instructions: "What colour was the ball?".into(),
                options: vec![
                    ("red".into(), String::new()),
                    ("blue".into(), String::new()),
                ],
            },
            Question::Score {
                instructions: "How long is the story?".into(),
                levels: vec![
                    "Very short".into(),
                    "Short".into(),
                    "Medium".into(),
                    "Long".into(),
                ],
            },
        ];
        // A tiny test model may not have every code as one token (stories260K
        // has no single-token Yes/No); ask what it can answer.
        let questions: Vec<Question> = all
            .into_iter()
            .filter(|q| model.codes.classes(q.kind(), q.n_classes()).is_ok())
            .collect();
        assert!(
            questions.len() >= 2,
            "the model can answer too few questions to compare"
        );

        let decide = |mode, content_free| {
            model
                .decide(state, &questions, DecideOptions { mode, content_free })
                .expect("decision")
        };
        let shared = decide(EvalMode::SharedPrefix, true);
        let separate = decide(EvalMode::Separate, false);
        eprintln!("shared:   {:?}", shared.stats);
        eprintln!("separate: {:?}", separate.stats);

        let mut worst: f64 = 0.0;
        for (i, (a, b)) in shared.outcomes.iter().zip(&separate.outcomes).enumerate() {
            eprintln!(
                "q{i}: shared {:?}\n    separate {:?}",
                a.logprobs, b.logprobs
            );
            for (x, y) in a.logprobs.iter().zip(&b.logprobs) {
                // A code the model all but rules out can sit at -30 in one
                // path and -31 in the other without meaning anything.
                if x.max(*y) > -15.0 {
                    worst = worst.max((x - y).abs());
                }
            }
            assert!(a.prior_logprobs.is_some() && b.prior_logprobs.is_none());
        }
        eprintln!("largest log-probability difference: {worst:.2e} (tolerance {tolerance:.0e})");
        assert!(
            worst < tolerance,
            "shared-prefix and separate disagree by {worst}"
        );
        assert!(shared.stats.shared_prefix_tokens > 0);
        assert!(shared.stats.evaluated_tokens < separate.stats.evaluated_tokens);
        assert_eq!(shared.stats.prompt_tokens, separate.stats.prompt_tokens);

        // Asked again, every prior comes from the cache.
        let again = decide(EvalMode::SharedPrefix, true);
        assert_eq!(again.priors_cached, questions.len());
        assert!(again.prior_stats.is_none());
    }
}
