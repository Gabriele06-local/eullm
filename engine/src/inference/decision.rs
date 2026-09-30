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
//! last. [`EvalMode::SharedPrefix`], the default, decodes the state's part
//! once on sequence 0, then each question on its own on sequence 1, which
//! starts as a copy of 0 (`copy_kv_cache_seq` on a unified cache only tags
//! the existing cells with the new sequence id — no data is copied) and is
//! removed again once the question's logits are read. For `Q` questions of
//! `q` tokens on a state of `S` that is about `S + Q·q` tokens instead of
//! the `Q·(S + q)` of asking them one at a time; and since every question
//! is decoded alone, in the same cells, its answer is the same whatever
//! else the request asks.
//!
//! [`EvalMode::Batched`] decodes the rest of every question in one batch
//! instead, one sequence each: fewer decode calls, but on quantized weights
//! a question's probabilities then move with the other questions in the
//! batch, by the model's own numerical noise. [`EvalMode::Separate`] asks
//! each question from an empty cache: the reference the other two are
//! tested against and the baseline they are benchmarked against.
//!
//! A separate model slot, like `embedding`, and for the same reason: nothing
//! here needs a sampler, a conversation-sized KV cache or the scheduler, and
//! a small decision model can stay resident next to the chat model.

mod engine;
mod verdict;

use std::collections::HashMap;
use std::path::Path;
use std::pin::pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token_type::LlamaTokenAttr;

/// Most questions one request may ask. Every question gets its own
/// vocabulary-sized logits row (~0.6 MB for a 150k vocabulary) and, in
/// `Batched` mode, its own sequence, which llama.cpp caps at 256 per
/// context; 64 is the top of the range the shared-prefix benchmark
/// measures.
pub const MAX_QUESTIONS: usize = 64;

/// Most options a `choice` question may have: the System One API's 255. A
/// code-readout model answers at most [`LETTER_CODES`] of them; a verdict
/// model reads every option at its own slot and has only its token budget.
pub const MAX_CHOICE_OPTIONS: usize = 255;

/// Options a code-readout model can answer: one letter each, `A`…`Z`.
/// Going past 26 needs multi-token codes or elimination rounds, which this
/// version does not do.
pub const LETTER_CODES: usize = 26;

/// Fewest options a `choice` question, or levels a `score` question, may
/// have.
pub const MIN_OPTIONS: usize = 2;

/// Most levels a `score` question may have: one digit each, `0`…`9` — the
/// same ceiling as the System One API.
pub const MAX_SCORE_LEVELS: usize = 10;

/// Default ceiling on the KV cells one request's context may use
/// (`--decision-ctx`). The context kept between requests grows to the
/// largest request so far, so this is not memory held from the start; it is
/// the most one request may ask for, and what the slot reserves VRAM for. A request needs the state plus its
/// longest question — plus every other question too in `Batched` mode — so
/// 8192 fits a state of nearly 8k tokens, or in `Batched` mode a 4k-token
/// state with about fifty short questions.
pub const DEFAULT_DECISION_CTX: u32 = 8192;

/// The state used to measure a question's content-free prior (Zhao et al.,
/// 2021, "Calibrate Before Use"): the same question asked about no content
/// at all shows which answer the model leans towards before reading
/// anything.
pub const CONTENT_FREE_STATE: &str = "N/A";

/// How `confidence` is computed ([`max_probability_confidence`]), reported
/// next to every answer and in every audit record so a client can tell
/// which definition a stored number came from. Until it was changed to
/// jev-style's it was [`ENTROPY_CONFIDENCE_METHOD`].
pub const CONFIDENCE_METHOD: &str = "normalized_max_probability";

/// The name of [`normalized_entropy_confidence`]: `confidence` in responses
/// and audit records written before [`CONFIDENCE_METHOD`] replaced it, and
/// `eullm.confidence_entropy` since.
pub const ENTROPY_CONFIDENCE_METHOD: &str = "normalized_entropy";

/// Most questions one `batched` decode round holds on a recurrent or
/// hybrid model, which keeps a recurrent state per sequence.
const RECURRENT_BATCH_GROUP: usize = 16;

/// The micro-batch Jev-Style's render-v1 scores were validated with.
const VERDICT_V1_UBATCH: usize = 1024;

/// llama.cpp attends over the cache's used cells rounded up to a multiple
/// of this, so the attention's shape moves in steps of it.
const KV_WINDOW_STEP: usize = 256;

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
        /// What an answer of true, and of false, means — empty when the
        /// question does not say (System One `criteria: {"true", "false"}`).
        true_means: String,
        false_means: String,
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

    /// A yes/no question with nothing said about what either answer means.
    pub fn noul(instructions: impl Into<String>) -> Self {
        Self::Noul {
            instructions: instructions.into(),
            true_means: String::new(),
            false_means: String::new(),
        }
    }

    pub fn instructions(&self) -> &str {
        match self {
            Self::Noul { instructions, .. }
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
            Self::Noul {
                true_means,
                false_means,
                ..
            } => {
                if true_means.contains('\0') || false_means.contains('\0') {
                    return Err("criteria must not contain NUL characters".to_string());
                }
                Ok(())
            }
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

/// What the user turn says before the state, and between the state and the
/// question. Everything up to the end of `QUESTION_LABEL` is the same for
/// every question about one state.
const STATE_LABEL: &str = "State:\n";
const QUESTION_LABEL: &str = "\n\nQuestion:";

/// The user turn for one question. The state comes first and the question
/// last on purpose: everything up to the question is shared by every
/// question about the same state, which is what [`EvalMode::SharedPrefix`]
/// decodes only once.
fn user_message(state: &str, question: &Question) -> String {
    format!(
        "{STATE_LABEL}{state}{QUESTION_LABEL}{}",
        question_text(question)
    )
}

/// The question's own part of the user turn, from the space after
/// `QUESTION_LABEL` to the last line.
fn question_text(question: &Question) -> String {
    use std::fmt::Write as _;

    let mut msg = format!(" {}\n", question.instructions().trim());
    match question {
        Question::Noul {
            true_means,
            false_means,
            ..
        } => {
            if !true_means.trim().is_empty() {
                let _ = writeln!(msg, "Yes means: {}", true_means.trim());
            }
            if !false_means.trim().is_empty() {
                let _ = writeln!(msg, "No means: {}", false_means.trim());
            }
            msg.push_str("Answer Yes or No.");
        }
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

/// How one model's prompts are put together, worked out once at load.
///
/// Rendering a prompt through the model's Jinja template costs ~9 ms each
/// for Qwen3's (measured: 598 ms for 64 questions, against 95 ms to decode
/// all of them on an RTX 5070 Ti), and tokenizing it costs time in
/// proportion to the state, once per question. Neither needs repeating
/// when the template embeds the user turn verbatim and the tokenizer
/// splits cleanly after `QUESTION_LABEL` — both checked here, so a model
/// where either does not hold falls back to the slow, always-correct path.
struct PromptLayout {
    /// The rendered text before and after the user turn: the template's own
    /// when it embeds the turn verbatim, the plain-text prompt's without a
    /// template. `None`: render every prompt.
    wrapper: Option<(String, String)>,
    /// `wrapper` cut at its control tokens, when tokenizing a prompt in
    /// those pieces was checked to give the tokens the whole prompt gives:
    /// a request's text is then never read as a control token. `None`: a
    /// request whose text contains one's is refused.
    cut: Option<TemplateCut>,
    /// Tokenizing everything up to `QUESTION_LABEL` and the rest apart
    /// gives exactly the whole prompt's tokens: the state is tokenized once
    /// per request instead of once per question.
    split_tokens: bool,
}

impl PromptLayout {
    fn resolve(model: &LlamaModel, uses_template: bool) -> Self {
        let tokenize = model_tokenizer(model);
        let wrapper = template_wrapper(|user| Some(render_prompt(model, uses_template, user)));
        let cut = wrapper.as_ref().and_then(|(head, tail)| {
            let cut = TemplateCut::new(head, tail, &control_texts(model, &[head, tail]));
            let holds = cut_tokenization_holds(&tokenize, head, tail, &cut);
            if !holds {
                tracing::warn!(
                    "Decision model: its template cannot be tokenized with a request's text kept \
                     apart from its control tokens; a request containing a control token's text \
                     will be refused"
                );
            }
            holds.then_some(cut)
        });
        let split_tokens = wrapper.as_ref().is_some_and(|(head, tail)| {
            split_tokenization_holds(&Wrapped {
                tokenize: &tokenize,
                head,
                tail,
                cut: cut.as_ref(),
            })
        });
        Self {
            wrapper,
            cut,
            split_tokens,
        }
    }

    fn describe(&self) -> &'static str {
        match (&self.wrapper, self.split_tokens) {
            (Some(_), true) => "template rendered once, state tokenized once per request",
            (Some(_), false) => "template rendered once, every prompt tokenized whole",
            (None, _) => "template rendered and tokenized per question",
        }
    }
}

/// Whether a tokenization reads the text of a control token — a chat
/// template's `<|im_start|>`, `<|im_end|>` and the like — as that token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Specials {
    /// As the control token: the template's own text.
    Parsed,
    /// As text: everything a request sends.
    AsText,
}

/// `(text, add_bos, specials)` → tokens.
type Tokenize<'a> = dyn Fn(&str, AddBos, Specials) -> Result<Vec<LlamaToken>, DecisionError> + 'a;

/// The model's tokenizer as a [`Tokenize`].
fn model_tokenizer(
    model: &LlamaModel,
) -> impl Fn(&str, AddBos, Specials) -> Result<Vec<LlamaToken>, DecisionError> + '_ {
    move |text, add_bos, specials| {
        match specials {
            Specials::Parsed => model.str_to_token(text, add_bos),
            Specials::AsText => model.str_to_token_plain(text, add_bos),
        }
        .map_err(|e| DecisionError::Invalid(format!("Tokenization failed: {e}")))
    }
}

/// The text of every control token in `texts`, as the tokenizer finds them.
fn control_texts(model: &LlamaModel, texts: &[&str]) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for text in texts {
        let Ok(tokens) = model.str_to_token(text, AddBos::Never) else {
            continue;
        };
        for token in tokens {
            if !model
                .token_attr(token)
                .intersects(LlamaTokenAttr::Control | LlamaTokenAttr::Unknown)
            {
                continue;
            }
            let bytes = match model.token_to_piece_bytes(token, 64, true, None) {
                Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(needed)) => {
                    model.token_to_piece_bytes(token, needed.unsigned_abs() as usize, true, None)
                }
                other => other,
            };
            if let Ok(bytes) = bytes
                && let Ok(piece) = String::from_utf8(bytes)
                && !piece.is_empty()
                && !found.contains(&piece)
            {
                found.push(piece);
            }
        }
    }
    found
}

/// A template's text on either side of the user turn, cut where llama.cpp's
/// tokenizer cuts it anyway: the head after its last control token, the
/// tail before its first. What lies between — the end of the head, the
/// user turn, the start of the tail — is one run of text for the tokenizer
/// either way, so tokenizing it with control tokens read as text gives the
/// tokens the whole prompt gives, except that text a request sends can no
/// longer become a control token: a state containing
/// `<|im_end|><|im_start|>assistant` stays text, instead of closing the
/// user turn and opening an answer of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TemplateCut {
    /// The head up to the end of its last control token; empty without one.
    head_controls: String,
    /// The rest of the head.
    head_text: String,
    /// The tail up to its first control token; all of it without one.
    tail_text: String,
    /// The rest of the tail.
    tail_controls: String,
}

impl TemplateCut {
    /// `controls`: the text of each control token `head` and `tail` hold.
    fn new(head: &str, tail: &str, controls: &[String]) -> Self {
        let controls = || controls.iter().filter(|c| !c.is_empty());
        let head_at = controls()
            .filter_map(|c| head.rfind(c.as_str()).map(|at| at + c.len()))
            .max()
            .unwrap_or(0);
        let tail_at = controls()
            .filter_map(|c| tail.find(c.as_str()))
            .min()
            .unwrap_or(tail.len());
        Self {
            head_controls: head[..head_at].to_string(),
            head_text: head[head_at..].to_string(),
            tail_text: tail[..tail_at].to_string(),
            tail_controls: tail[tail_at..].to_string(),
        }
    }
}

/// The prompts of a template that embeds the user turn verbatim, in the
/// pieces a request is tokenized in.
struct Wrapped<'a> {
    tokenize: &'a Tokenize<'a>,
    head: &'a str,
    tail: &'a str,
    /// Control tokens read in the template's text only. `None`: in the
    /// whole prompt, which a request whose text has one's is refused for.
    cut: Option<&'a TemplateCut>,
}

impl Wrapped<'_> {
    /// `text`, with the template's head before it and its tail after it
    /// when asked for.
    fn tokens(&self, text: &str, head: bool, tail: bool) -> Result<Vec<LlamaToken>, DecisionError> {
        let pick = |on: bool, piece: &str| if on { piece.to_string() } else { String::new() };
        let bos = if head { AddBos::Always } else { AddBos::Never };
        let Some(cut) = self.cut else {
            let whole = format!("{}{text}{}", pick(head, self.head), pick(tail, self.tail));
            return (self.tokenize)(&whole, bos, Specials::Parsed);
        };
        let mut tokens = if head {
            (self.tokenize)(&cut.head_controls, bos, Specials::Parsed)?
        } else {
            Vec::new()
        };
        let text = format!(
            "{}{text}{}",
            pick(head, &cut.head_text),
            pick(tail, &cut.tail_text)
        );
        tokens.extend((self.tokenize)(&text, AddBos::Never, Specials::AsText)?);
        if tail {
            tokens.extend((self.tokenize)(
                &cut.tail_controls,
                AddBos::Never,
                Specials::Parsed,
            )?);
        }
        Ok(tokens)
    }

    /// The whole prompt around one user turn.
    fn whole(&self, user: &str) -> Result<Vec<LlamaToken>, DecisionError> {
        self.tokens(user, true, true)
    }

    /// Everything every question about `state` starts with, up to the end
    /// of `QUESTION_LABEL`.
    fn common(&self, state: &str) -> Result<Vec<LlamaToken>, DecisionError> {
        self.tokens(
            &format!("{STATE_LABEL}{state}{QUESTION_LABEL}"),
            true,
            false,
        )
    }

    /// One question's own text, then the template's tail.
    fn rest(&self, question: &str) -> Result<Vec<LlamaToken>, DecisionError> {
        self.tokens(question, false, true)
    }
}

/// Whether tokenizing prompts in the pieces of `cut` gives exactly the
/// tokens of the whole prompt, for user turns without control-token text:
/// a template whose control tokens strip the whitespace next to them, or a
/// tokenizer that ends every call with a token of its own, fails here.
fn cut_tokenization_holds(
    tokenize: &Tokenize<'_>,
    head: &str,
    tail: &str,
    cut: &TemplateCut,
) -> bool {
    let with = Wrapped {
        tokenize,
        head,
        tail,
        cut: Some(cut),
    };
    let without = Wrapped { cut: None, ..with };
    let same =
        |a: Result<Vec<LlamaToken>, DecisionError>, b| matches!((a, b), (Ok(a), Ok(b)) if a == b);
    // A probe that itself holds a control token's text would, rightly,
    // come out different: it proves nothing either way.
    let clean = PROBES.iter().filter(|probe| {
        same(
            tokenize(probe, AddBos::Never, Specials::Parsed),
            tokenize(probe, AddBos::Never, Specials::AsText),
        )
    });
    let mut checked = 0;
    for probe in clean {
        if !same(with.whole(probe), without.whole(probe)) {
            return false;
        }
        checked += 1;
    }
    checked > 0
}

/// Refuses a request whose text the tokenizer would read as a control
/// token, for a model whose prompts cannot be tokenized with that text kept
/// as text: let through, `<|im_end|>` in a state would end the user turn
/// there and let the state write the rest of the prompt.
fn refuse_control_text(
    tokenize: &Tokenize<'_>,
    state: &str,
    questions: &[Question],
) -> Result<(), DecisionError> {
    // The state first, then question `i` as text `i + 1`.
    let texts = std::iter::once(state.to_string()).chain(questions.iter().map(question_text));
    for (i, text) in texts.enumerate() {
        if tokenize(&text, AddBos::Never, Specials::Parsed)?
            != tokenize(&text, AddBos::Never, Specials::AsText)?
        {
            let refused = DecisionError::Invalid(
                "the request contains the text of one of this model's control tokens (a chat \
                 template's turn marker, for instance), and this model's template cannot be \
                 tokenized with it kept as text: refused rather than let it change where the \
                 prompt's turns begin and end"
                    .to_string(),
            );
            return Err(match i.checked_sub(1) {
                Some(question) => DecisionError::Question(question, Box::new(refused)),
                None => refused,
            });
        }
    }
    Ok(())
}

/// User turns that carry what a real one can: newlines, brackets and
/// quotes, markup, a `</think>`, non-ASCII text.
const PROBES: [&str; 2] = [
    "State:\nx {a} [b] <c> \"d\" 'e' </think>\n\nQuestion: Is it?\nA) f: g\nAnswer Yes or No.",
    "State:\n«Il cliente» chiede — €12,50\n\nQuestion: Qual è?\n0) basso\n1) alto\nReply.",
];

/// The text a rendered prompt puts before and after the user turn, when
/// every probe comes back verbatim between the same two strings. The probes
/// carry what a real turn can: newlines, brackets and quotes, markup, a
/// `</think>`, non-ASCII text — a template that escapes, trims inside or
/// rewrites any of it fails the check and keeps per-question rendering.
fn template_wrapper<F>(render: F) -> Option<(String, String)>
where
    F: Fn(&str) -> Option<String>,
{
    let mut found: Option<(String, String)> = None;
    for probe in PROBES {
        let rendered = render(probe)?;
        let at = rendered.find(probe)?;
        let (head, tail) = (&rendered[..at], &rendered[at + probe.len()..]);
        match &found {
            None => found = Some((head.to_string(), tail.to_string())),
            Some((h, t)) if h == head && t == tail => {}
            Some(_) => return None,
        }
    }
    found
}

/// Whether tokenizing `head + state + QUESTION_LABEL` and `question + tail`
/// apart gives exactly the tokens of the whole prompt, for questions that
/// start with any of a spread of characters. The split falls after the
/// label's colon; a tokenizer that merges across it (`": I"` as one token),
/// or that prefixes a fragment with a space of its own (SentencePiece's
/// dummy prefix), fails here and keeps whole-prompt tokenization.
fn split_tokenization_holds(wrapped: &Wrapped<'_>) -> bool {
    const STARTS: [&str; 10] = [
        "Is it urgent?",
        "is it",
        "3 items",
        "«Quoted»",
        "\"Quoted\"",
        "(a)",
        "Qual è l'area?",
        "¿Qué?",
        "什么",
        "-x",
    ];
    let Ok(common_tokens) = wrapped.common("x") else {
        return false;
    };
    STARTS.iter().all(|start| {
        let question = format!(" {start}\nAnswer Yes or No.");
        match (
            wrapped.whole(&format!("{STATE_LABEL}x{QUESTION_LABEL}{question}")),
            wrapped.rest(&question),
        ) {
            (Ok(whole), Ok(rest_tokens)) => {
                whole.len() == common_tokens.len() + rest_tokens.len()
                    && whole[..common_tokens.len()] == common_tokens[..]
                    && whole[common_tokens.len()..] == rest_tokens[..]
            }
            _ => false,
        }
    })
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
            Self::Letters => LETTER_CODES,
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
        if n > set.len() {
            return Err(format!(
                "this decision model answers {} questions with at most {} classes, one {} code \
                 each; got {n}",
                kind.as_str(),
                set.len(),
                set.name()
            ));
        }
        let classes = &self.sets[set.index()][..n];
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

/// A cached content-free prior: the mode it was measured in, and its prompt.
type PriorKey = (EvalMode, Vec<LlamaToken>);

/// How the prompts of one request are evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvalMode {
    /// Decode the state once, then every question on its own in the cells
    /// right after it — the same cells, the same batch shape, whatever else
    /// the request asks. A question's probabilities depend on the state and
    /// that question only. The default.
    SharedPrefix,
    /// Decode the prefix every prompt shares once, then the rest of every
    /// prompt in one batch: the fewest decode calls, but a question's
    /// probabilities move with the other questions in the batch, by the
    /// model's own numerical noise — up to 0.5 in probability on Qwen3-0.6B
    /// on an RTX 5070 Ti, measured.
    Batched,
    /// Decode every prompt on its own from an empty cache: the baseline.
    Separate,
}

impl EvalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SharedPrefix => "shared_prefix",
            Self::Batched => "batched",
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
    /// Tokens decoded once for every prompt: the state's part of them for
    /// `SharedPrefix`, all they start with in common for `Batched`, none
    /// for `Separate`.
    pub shared_prefix_tokens: usize,
    /// Tokens actually decoded.
    pub evaluated_tokens: usize,
    /// Sum of the prompts' lengths: what decoding each on its own costs.
    pub prompt_tokens: usize,
    /// KV cells of the context the request ran in: the one kept from
    /// earlier requests when it was large enough.
    pub context_cells: usize,
    /// Creating that context; 0 when it was kept.
    pub context_ms: f64,
    /// Decoding the shared prefix (not `Separate`); 0 when it was reused.
    pub prefix_ms: f64,
    /// The shared prefix was already in the context, left there by the
    /// previous request, and was not decoded again.
    pub prefix_reused: bool,
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

/// Set when the request a decision is for is abandoned — its client has
/// disconnected — so the work stops instead of occupying the model to the
/// end while the next request waits behind it. Clones share one flag.
///
/// It is read where the kept context is in a state the next request can
/// build on: before a request starts (a request that waited for the model
/// while its client left never starts), between the calls that decode the
/// state, and between questions — never in the middle of one, whose cells
/// would be left behind in the next one's way. A request stopped this way
/// ends in [`DecisionError::Cancelled`], with no answers.
#[derive(Debug, Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// `Err(Cancelled)` once the request has been abandoned.
    fn check(&self) -> Result<(), DecisionError> {
        if self.is_cancelled() {
            return Err(DecisionError::Cancelled);
        }
        Ok(())
    }
}

/// Why a decision could not be made. The split is the one the API maps to
/// error codes: `Invalid` is a request the client must change, `TooLong`
/// and `OverBudget` one too long to read whole, `Runtime` our failure; and
/// `Question` says which question it was, when it was one question's.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionError {
    /// Not answerable as asked: a limit, an empty field, a code this model's
    /// tokenizer cannot give in one token.
    Invalid(String),
    /// More KV cells than one request may use.
    TooLong { needed: usize, limit: usize },
    /// More tokens than the model itself reads: a Jev-Style model's input
    /// budgets, or the context it was trained on. Refused, never truncated.
    OverBudget(String),
    /// The error of the question at this index, in the order the request
    /// asked them.
    Question(usize, Box<DecisionError>),
    /// The request was abandoned ([`Cancel`]) and stopped before its
    /// answers were complete.
    Cancelled,
    /// llama.cpp failed.
    Runtime(String),
}

impl DecisionError {
    /// This error, as the one of question `index`.
    fn in_question(index: usize) -> impl FnOnce(Self) -> Self {
        move |e| Self::Question(index, Box::new(e))
    }
}

impl std::fmt::Display for DecisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(msg) | Self::OverBudget(msg) | Self::Runtime(msg) => f.write_str(msg),
            Self::TooLong { needed, limit } => write!(
                f,
                "this request needs {needed} tokens of context but the decision model allows \
                 {limit} per request (--decision-ctx): shorten the state or the longest \
                 question, or in \"batched\" mode ask fewer questions at once"
            ),
            Self::Question(_, e) => e.fmt(f),
            Self::Cancelled => f.write_str(
                "the request was abandoned by its client and stopped before its answers were \
                 complete",
            ),
        }
    }
}

impl std::error::Error for DecisionError {}

fn runtime(what: &str, e: impl std::fmt::Display) -> DecisionError {
    DecisionError::Runtime(format!("{what}: {e}"))
}

/// How a decision model's answers are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadoutKind {
    /// The next-token distribution after a chat prompt, restricted to
    /// single-token answer codes: any instruction-tuned model.
    Codes,
    /// `logit(" yes") - logit(" no")` at a slot after every option: a
    /// Jev-Style model, trained for it (`decision::verdict`).
    Verdict,
}

impl ReadoutKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Codes => "codes",
            Self::Verdict => "verdict",
        }
    }
}

/// What a code-readout model's prompts are built with.
struct CodeReadout {
    uses_template: bool,
    layout: PromptLayout,
    codes: CodeTable,
}

enum ModelReadout {
    Codes(CodeReadout),
    Verdict(verdict::VerdictModel),
}

/// A loaded decision model: a `LlamaModel` on the process-wide shared
/// backend (see `EmbeddingModel::load` for why it must be shared), how its
/// answers are read — the code table resolved against its tokenizer, or a
/// verdict model's protocol — the worker that evaluates its prompts
/// (`engine`), and a cache of content-free priors.
pub struct DecisionModel {
    model: Arc<LlamaModel>,
    /// `--no-flash-attn` turned off, as for the generation model: llama.cpp
    /// then decides per device (its AUTO policy); on, flash attention is
    /// never used.
    flash_attn: bool,
    readout: ModelReadout,
    /// Held for a whole request, so the priors and the answers of one
    /// request reach the worker back to back and the state it keeps is the
    /// request's own.
    eval_lock: Mutex<()>,
    /// Keyed by mode as well as prompt: a prior measured in one mode is not
    /// bit-for-bit the one another mode would measure.
    prior_cache: Mutex<HashMap<PriorKey, Vec<f64>>>,
    /// F32 KV cache and no flash attention: takes the cache's own rounding
    /// out of the comparison between shared-prefix and separate decoding,
    /// so on an F32 model the equivalence test checks the logic to ~1e-6
    /// instead of to the model's numerical noise. Only the tests set it; it
    /// costs twice the KV memory.
    exact: bool,
    engine: engine::Engine,
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
        flash_attn: bool,
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

        let flash = if flash_attn { "auto" } else { "off" };
        let (readout, protocol) = match verdict::VerdictModel::detect(&model, path)? {
            Some(v) => {
                tracing::info!(
                    "Decision model loaded — {}; flash attention {flash}; up to {max_ctx} tokens \
                     of context per request",
                    v.describe()
                );
                let protocol = match v.render {
                    // The ubatch its scores were validated with.
                    verdict::Render::V1 => engine::Protocol {
                        batch: 4096,
                        ubatch: VERDICT_V1_UBATCH as u32,
                        blocks: false,
                        group: RECURRENT_BATCH_GROUP,
                        cpu_flash_attn_off: false,
                    },
                    verdict::Render::V2 => engine::Protocol {
                        batch: verdict::BLOCK_V2 as u32,
                        ubatch: verdict::BLOCK_V2 as u32,
                        blocks: true,
                        group: 1,
                        cpu_flash_attn_off: true,
                    },
                };
                (ModelReadout::Verdict(v), protocol)
            }
            None => {
                let uses_template = super::render_jinja_chat_template(
                    &model,
                    &[("system", SYSTEM_PROMPT), ("user", "x")],
                    false,
                )
                .is_some();
                if !uses_template {
                    tracing::warn!(
                        "Decision model has no usable chat template — using a plain-text \
                         prompt, which instruction-tuned models follow less reliably"
                    );
                }
                let codes = CodeTable::resolve(&model, uses_template)?;
                let layout = PromptLayout::resolve(&model, uses_template);
                tracing::info!(
                    "Decision model loaded — codes: {}; prompts: {}; flash attention {flash}; up \
                     to {max_ctx} tokens of context per request",
                    codes.summary(),
                    layout.describe(),
                );
                let mut protocol = engine::Protocol::CODES;
                if model.is_recurrent() || model.is_hybrid() {
                    // A recurrent layer keeps a state per sequence: 64 of
                    // them for one `batched` round would cost more than the
                    // round saves.
                    protocol.group = RECURRENT_BATCH_GROUP;
                }
                let code = CodeReadout {
                    uses_template,
                    layout,
                    codes,
                };
                (ModelReadout::Codes(code), protocol)
            }
        };

        let threads = threads.max(1);
        let model = Arc::new(model);
        let engine = engine::Engine::start(
            Arc::clone(&model),
            backend,
            engine::EngineConfig {
                threads,
                flash_attn,
                max_ctx,
                on_gpu: gpu_layers != 0,
                protocol,
            },
        )
        .map_err(|e| format!("Failed to start the decision worker: {e}"))?;

        Ok(Self {
            model,
            flash_attn,
            readout,
            eval_lock: Mutex::new(()),
            prior_cache: Mutex::new(HashMap::new()),
            exact: false,
            engine,
        })
    }

    /// Answer every question about `state`: per class per question, the
    /// full-vocabulary log-probability of its code — and optionally the same
    /// measured on [`CONTENT_FREE_STATE`] — or, for a verdict model, its
    /// option's score. Stops with [`DecisionError::Cancelled`] at the next
    /// question once `cancel` is set.
    pub fn decide(
        &self,
        state: &str,
        questions: &[Question],
        options: DecideOptions,
        cancel: &Cancel,
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
        for (i, question) in questions.iter().enumerate() {
            question
                .validate()
                .map_err(DecisionError::Invalid)
                .map_err(DecisionError::in_question(i))?;
        }
        let code = match &self.readout {
            ModelReadout::Codes(code) => code,
            ModelReadout::Verdict(v) => {
                return self.decide_verdict(v, state, questions, options, cancel);
            }
        };
        let mut classes = Vec::with_capacity(questions.len());
        for (i, question) in questions.iter().enumerate() {
            classes.push(
                code.codes
                    .classes(question.kind(), question.n_classes())
                    .map_err(DecisionError::Invalid)
                    .map_err(DecisionError::in_question(i))?,
            );
        }
        let (prompts, state_prefix) = self.prompts(code, state, questions)?;

        let _running = self
            .eval_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // Abandoned while it waited for the request before it.
        cancel.check()?;

        // The priors first: the worker keeps the state it decoded last, and
        // the next request is far likelier to ask about this one than about
        // `CONTENT_FREE_STATE`.
        let mut priors: Option<Vec<Vec<f64>>> = None;
        let mut prior_stats = None;
        let mut priors_cached = 0;
        if options.content_free {
            let (cf_prompts, cf_prefix) = self.prompts(code, CONTENT_FREE_STATE, questions)?;
            let mut found: Vec<Option<Vec<f64>>> = {
                let cache = self
                    .prior_cache
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                cf_prompts
                    .iter()
                    .map(|p| cache.get(&(options.mode, p.clone())).cloned())
                    .collect()
            };
            priors_cached = found.iter().filter(|p| p.is_some()).count();
            let missing: Vec<usize> = (0..found.len()).filter(|&i| found[i].is_none()).collect();
            if !missing.is_empty() {
                let missing_prompts: Vec<Vec<LlamaToken>> =
                    missing.iter().map(|&i| cf_prompts[i].clone()).collect();
                let missing_classes: Vec<&[Vec<LlamaToken>]> =
                    missing.iter().map(|&i| classes[i]).collect();
                let (measured, cf_stats) = self.evaluate(
                    missing_prompts,
                    cf_prefix,
                    &missing_classes,
                    options.mode,
                    cancel,
                )?;
                let mut cache = self
                    .prior_cache
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                if cache.len() + missing.len() > MAX_PRIOR_CACHE {
                    cache.clear();
                }
                for (&i, lp) in missing.iter().zip(measured) {
                    cache.insert((options.mode, cf_prompts[i].clone()), lp.clone());
                    found[i] = Some(lp);
                }
                prior_stats = Some(cf_stats);
            }
            priors = Some(found.into_iter().map(Option::unwrap_or_default).collect());
        }

        let (logprobs, stats) =
            self.evaluate(prompts, state_prefix, &classes, options.mode, cancel)?;

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

    /// A verdict model's questions, all in one job.
    fn decide_verdict(
        &self,
        v: &verdict::VerdictModel,
        state: &str,
        questions: &[Question],
        options: DecideOptions,
        cancel: &Cancel,
    ) -> Result<Decision, DecisionError> {
        if options.content_free {
            return Err(DecisionError::Invalid(format!(
                "content_free calibration is for code-readout models; {} was released with a \
                 calibration temperature of its own, applied by default (\"temperature\" \
                 overrides it, 1 turns it off)",
                v.name
            )));
        }
        let encode = |text: &str| {
            self.model
                .str_to_token_plain(text, AddBos::Never)
                .map_err(|e| DecisionError::Invalid(format!("Tokenization failed: {e}")))
        };
        let prefix = verdict::prefix(&encode, state)?;
        let rendered = questions
            .iter()
            .enumerate()
            .map(|(i, q)| {
                verdict::render(v, &encode, &prefix, q).map_err(DecisionError::in_question(i))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let p = prefix.len();
        // The whole state is decoded once. With render v2, whose input is
        // cut into blocks at the state's end anyway, that changes no score.
        // With render v1 it moves them by the model's own noise, since
        // llama.cpp's kernels and the recurrent layers' chunking depend on
        // where a micro-batch starts: up to 0.04 in probability, measured on
        // a CPU and on CUDA.
        // `separate` gives the release runtime's scores exactly.
        let shared = match options.mode {
            EvalMode::Separate => 0,
            EvalMode::SharedPrefix | EvalMode::Batched => p,
        };
        let _running = self
            .eval_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        cancel.check()?;
        let (scores, stats) = self.engine.evaluate(engine::Job {
            reads: rendered.iter().map(|r| r.slots.clone()).collect(),
            blocks: rendered.iter().map(|r| r.blocks.clone()).collect(),
            prompts: rendered.into_iter().map(|r| r.ids).collect(),
            shared,
            readout: engine::Readout::Verdict {
                yes: v.yes,
                no: v.no,
            },
            mode: options.mode,
            exact: self.exact,
            cancel: cancel.clone(),
        })?;
        let outcomes = scores
            .into_iter()
            .zip(questions)
            .map(|(mut scores, question)| {
                // The model reads false before true; the classes are yes, no.
                if question.kind() == QuestionKind::Noul {
                    scores.reverse();
                }
                QuestionOutcome {
                    logprobs: scores,
                    prior_logprobs: None,
                }
            })
            .collect();
        Ok(Decision {
            outcomes,
            stats,
            prior_stats: None,
            priors_cached: 0,
        })
    }

    /// Whether flash attention may be used (`--no-flash-attn` not given).
    pub fn flash_attn(&self) -> bool {
        self.flash_attn
    }

    /// How this model's answers are read.
    pub fn readout(&self) -> ReadoutKind {
        match self.readout {
            ModelReadout::Codes(_) => ReadoutKind::Codes,
            ModelReadout::Verdict(_) => ReadoutKind::Verdict,
        }
    }

    /// The temperature probabilities are scaled with unless a request says
    /// otherwise: a verdict model's own calibration, 1 (none) otherwise.
    pub fn default_temperature(&self) -> f64 {
        match &self.readout {
            ModelReadout::Codes(_) => 1.0,
            ModelReadout::Verdict(v) => v.temperature,
        }
    }

    /// Free the context the worker keeps between requests until the next
    /// request needs one. A generation model is sized with this model's
    /// context counted as reserved (`fit::decision_reserve_bytes`), so it
    /// must not also be counted as used.
    pub fn release_context(&self) {
        self.engine.release();
    }

    /// Every question's prompt, and how many tokens at their start depend on
    /// the state alone: the template's head, the state and the question
    /// label. `SharedPrefix` decodes exactly those once, so the boundary
    /// between shared and per-question tokens never moves with the set of
    /// questions asked.
    fn prompts(
        &self,
        code: &CodeReadout,
        state: &str,
        questions: &[Question],
    ) -> Result<(Vec<Vec<LlamaToken>>, usize), DecisionError> {
        let tokenize = model_tokenizer(&self.model);
        let wrapped = code.layout.wrapper.as_ref().map(|(head, tail)| Wrapped {
            tokenize: &tokenize,
            head,
            tail,
            cut: code.layout.cut.as_ref(),
        });
        if wrapped.as_ref().is_none_or(|w| w.cut.is_none()) {
            refuse_control_text(&tokenize, state, questions)?;
        }
        let (prompts, common) = match &wrapped {
            Some(wrapped) if code.layout.split_tokens => {
                let common = wrapped.common(state)?;
                let prompts = questions
                    .iter()
                    .map(|q| {
                        let rest = wrapped.rest(&question_text(q))?;
                        let mut tokens = Vec::with_capacity(common.len() + rest.len());
                        tokens.extend_from_slice(&common);
                        tokens.extend(rest);
                        Ok(tokens)
                    })
                    .collect::<Result<Vec<_>, DecisionError>>()?;
                // The load-time check covers the characters a question can
                // start with; this one covers the request at hand, in debug
                // builds, where the extra whole-prompt tokenization is free.
                debug_assert!(questions.first().is_none_or(|q| {
                    wrapped
                        .whole(&user_message(state, q))
                        .is_ok_and(|whole| whole == prompts[0])
                }));
                let n = common.len();
                return Ok((prompts, n));
            }
            Some(wrapped) => {
                let prompts = questions
                    .iter()
                    .map(|q| wrapped.whole(&user_message(state, q)))
                    .collect::<Result<Vec<_>, DecisionError>>()?;
                (prompts, wrapped.common(state)?)
            }
            None => {
                let render = |q: &Question| {
                    tokenize(
                        &render_prompt(&self.model, code.uses_template, &user_message(state, q)),
                        AddBos::Always,
                        Specials::Parsed,
                    )
                };
                let prompts = questions
                    .iter()
                    .map(render)
                    .collect::<Result<Vec<_>, _>>()?;
                // Without the template's head and tail as text, the state's
                // part is what two unrelated questions about it share.
                let a = render(&Question::noul("Alpha?"))?;
                let b = render(&Question::noul("7?"))?;
                let common = a[..common_prefix(&a, &b)].to_vec();
                (prompts, common)
            }
        };
        // Tokenized apart from what follows it, the state's part can end in
        // a token or two the whole prompt merges differently: step back past
        // them, then check every prompt really starts with what is left.
        let boundary = common.len().saturating_sub(2);
        let state_prefix = if prompts
            .iter()
            .all(|p| p.len() > boundary && p[..boundary] == common[..boundary])
        {
            boundary
        } else {
            let refs: Vec<&[LlamaToken]> = prompts.iter().map(Vec::as_slice).collect();
            shared_prefix_len(&refs)
        };
        Ok((prompts, state_prefix))
    }

    fn evaluate(
        &self,
        prompts: Vec<Vec<LlamaToken>>,
        state_prefix: usize,
        classes: &[&[Vec<LlamaToken>]],
        mode: EvalMode,
        cancel: &Cancel,
    ) -> Result<(Vec<Vec<f64>>, EvalStats), DecisionError> {
        let n_ctx_train = self.model.n_ctx_train() as usize;
        if let Some(longest) = prompts.iter().map(Vec::len).max()
            && n_ctx_train > 0
            && longest > n_ctx_train
        {
            return Err(DecisionError::OverBudget(format!(
                "a prompt of {longest} tokens is longer than the {n_ctx_train} this model was \
                 trained on: shorten the state"
            )));
        }
        let shortest = prompts.iter().map(Vec::len).min().unwrap_or(0);
        let shared = match mode {
            // Every prompt keeps at least its last token, the one read.
            EvalMode::SharedPrefix => state_prefix.min(shortest.saturating_sub(1)),
            EvalMode::Batched => {
                let refs: Vec<&[LlamaToken]> = prompts.iter().map(Vec::as_slice).collect();
                shared_prefix_len(&refs)
            }
            EvalMode::Separate => 0,
        };
        let reads = prompts.iter().map(|p| vec![p.len() - 1]).collect();
        self.engine.evaluate(engine::Job {
            prompts,
            reads,
            shared,
            blocks: Vec::new(),
            readout: engine::Readout::Classes(classes.iter().map(|c| c.to_vec()).collect()),
            mode,
            exact: self.exact,
            cancel: cancel.clone(),
        })
    }
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// How many tokens `a` and `b` start with in common.
fn common_prefix(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
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

/// Per-class log-probabilities of logits rows, as `(prompt, logprobs)`, on
/// up to `workers` threads: each row is a pass over the whole vocabulary
/// (~0.5-0.9 ms for 150k entries on one core, measured), which for 64
/// questions would otherwise add tens of milliseconds to a batch the GPU
/// finished long before.
fn score_rows(
    rows: &[(usize, &[f32])],
    classes: &[&[Vec<LlamaToken>]],
    workers: usize,
) -> Result<Vec<(usize, Vec<f64>)>, DecisionError> {
    let score = |part: &[(usize, &[f32])]| -> Result<Vec<(usize, Vec<f64>)>, DecisionError> {
        part.iter()
            .map(|&(prompt, logits)| class_logprobs(logits, classes[prompt]).map(|lp| (prompt, lp)))
            .collect()
    };
    let workers = workers.clamp(1, rows.len().max(1));
    if workers == 1 {
        return score(rows);
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

/// `(K · p_max − 1) / (K − 1)`, clipped to [0, 1]: the top probability's
/// lead over an even split, as a share of the most it could lead by — 1
/// when all probability is on one class, 0 when it is spread evenly over
/// all `K`. It is jev-style's definition (`adapter.confidence`), for a
/// `choice` and a `score` alike, so the number means the same whether a
/// client asks jev-style's server or EuLLM; the System One API only says
/// confidence is "derived from the answer's probability distribution".
/// With two classes it is `2 · p_max − 1`; with one, 1, as jev-style has
/// it for a one-option `choice`.
pub fn max_probability_confidence(probabilities: &[f64]) -> f64 {
    let k = probabilities.len();
    if k <= 1 {
        return 1.0;
    }
    let top = probabilities.iter().copied().fold(0.0, f64::max);
    let k = k as f64;
    ((k * top - 1.0) / (k - 1.0)).clamp(0.0, 1.0)
}

/// `1 − H(p) / ln K`: 1 when all probability is on one class, 0 when it is
/// spread evenly over all `K`. `confidence` until jev-style's
/// [`max_probability_confidence`] replaced it, and still reported next to
/// it: it reads every class's probability, not only the top one, so two
/// answers with the same top probability but a different runner-up tell
/// apart.
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
        for confidence in [max_probability_confidence, normalized_entropy_confidence] {
            assert!(close(confidence(&[1.0, 0.0, 0.0]), 1.0));
            assert!(close(confidence(&[0.25; 4]), 0.0));
            assert!(close(confidence(&[0.5, 0.5]), 0.0));
        }
        let c = normalized_entropy_confidence(&[0.9, 0.1]);
        assert!(c > 0.5 && c < 1.0);
    }

    /// The values jev-style's `adapter.confidence` gives, edge cases
    /// included.
    #[test]
    fn confidence_is_jev_styles() {
        // Two classes: 2 · p_max − 1.
        assert!(close(max_probability_confidence(&[0.9, 0.1]), 0.8));
        assert!(close(max_probability_confidence(&[0.25, 0.75]), 0.5));
        // (3 · 0.6 − 1) / 2.
        assert!(close(max_probability_confidence(&[0.6, 0.3, 0.1]), 0.4));
        assert!(close(max_probability_confidence(&[0.1, 0.8, 0.1]), 0.7));
        // One class, or none: certain.
        assert_eq!(max_probability_confidence(&[1.0]), 1.0);
        assert_eq!(max_probability_confidence(&[]), 1.0);
        // Probabilities that sum to less than one cannot go below 0, nor
        // rounding above 1.
        assert_eq!(max_probability_confidence(&[0.2, 0.2, 0.2]), 0.0);
        assert_eq!(max_probability_confidence(&[1.0 + 1e-12, 0.0]), 1.0);
        // It reads the top probability only; the entropy tells a close
        // runner-up from an even rest.
        let (a, b) = ([0.6, 0.4, 0.0], [0.6, 0.2, 0.2]);
        assert!(close(
            max_probability_confidence(&a),
            max_probability_confidence(&b)
        ));
        assert!(normalized_entropy_confidence(&a) > normalized_entropy_confidence(&b));
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
        let q1 = Question::noul("Is it urgent?");
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
    fn a_template_that_embeds_the_turn_verbatim_is_rendered_once() {
        let render = |user: &str| {
            Some(format!(
                "<s>system\nSYS</s><s>user\n{user}</s><s>assistant\n"
            ))
        };
        let (head, tail) = template_wrapper(render).expect("verbatim template");
        assert_eq!(head, "<s>system\nSYS</s><s>user\n");
        assert_eq!(tail, "</s><s>assistant\n");
    }

    #[test]
    fn a_template_that_rewrites_the_turn_is_rendered_per_question() {
        // Escapes markup: the turn no longer appears verbatim.
        let escaping = |user: &str| Some(format!("[{}]", user.replace('<', "&lt;")));
        assert!(template_wrapper(escaping).is_none());
        // Wraps it in something that depends on the turn itself.
        let counting = |user: &str| Some(format!("{} chars: {user}", user.len()));
        assert!(template_wrapper(counting).is_none());
        // Does not render at all.
        assert!(template_wrapper(|_: &str| None).is_none());
    }

    /// `split_tokenization_holds` for a template `H…T` and a tokenizer
    /// `(text, add_bos) -> ids`.
    fn splits<F: Fn(&str, bool) -> Option<Vec<i32>>>(tokenize: F) -> bool {
        let tokenize = move |text: &str,
                             add_bos: AddBos,
                             _: Specials|
              -> Result<Vec<LlamaToken>, DecisionError> {
            Ok(tokenize(text, matches!(add_bos, AddBos::Always))
                .unwrap()
                .into_iter()
                .map(LlamaToken)
                .collect())
        };
        split_tokenization_holds(&Wrapped {
            tokenize: &tokenize,
            head: "H",
            tail: "T",
            cut: None,
        })
    }

    #[test]
    fn split_tokenization_holds_only_where_the_boundary_is_a_clean_split() {
        // One token per character: splitting anywhere is exact.
        let chars = |text: &str, _bos: bool| Some(text.chars().map(|c| c as i32).collect());
        assert!(splits(chars));

        // A token that spans the label's colon and the question's first
        // word, as a BPE merge across the boundary would.
        let merging = |text: &str, _bos: bool| {
            let mut out = Vec::new();
            let mut rest = text;
            while let Some(c) = rest.chars().next() {
                if rest.starts_with(": I") {
                    out.push(-1);
                    rest = &rest[3..];
                } else {
                    out.push(c as i32);
                    rest = &rest[c.len_utf8()..];
                }
            }
            Some(out)
        };
        assert!(!splits(merging));

        // A marker in front of every fragment, as SentencePiece's dummy
        // prefix space adds.
        let prefixing = |text: &str, _bos: bool| {
            Some(
                std::iter::once(0)
                    .chain(text.chars().map(|c| c as i32))
                    .collect(),
            )
        };
        assert!(!splits(prefixing));
    }

    /// A tokenizer with one control token, `<c>` (id 0 when parsed), and
    /// one token per character otherwise; BOS is id 1.
    fn tokenize_with_control(
        text: &str,
        add_bos: AddBos,
        specials: Specials,
    ) -> Result<Vec<LlamaToken>, DecisionError> {
        let mut out = Vec::new();
        if matches!(add_bos, AddBos::Always) {
            out.push(LlamaToken(1));
        }
        let mut rest = text;
        while let Some(c) = rest.chars().next() {
            if specials == Specials::Parsed && rest.starts_with("<c>") {
                out.push(LlamaToken(0));
                rest = &rest[3..];
            } else {
                out.push(LlamaToken(c as i32));
                rest = &rest[c.len_utf8()..];
            }
        }
        Ok(out)
    }

    #[test]
    fn a_template_is_cut_at_its_outermost_control_tokens() {
        let controls = ["<c>".to_string(), "<d>".to_string()];
        let cut = TemplateCut::new("<c>sys<d>\n<c>user\n", "<d>\n<c>bot\n", &controls);
        assert_eq!(cut.head_controls, "<c>sys<d>\n<c>");
        assert_eq!(cut.head_text, "user\n");
        assert_eq!(cut.tail_text, "");
        assert_eq!(cut.tail_controls, "<d>\n<c>bot\n");
        // Without control tokens, all of it is text.
        let plain = TemplateCut::new("System: ", "\nAnswer:", &[]);
        assert_eq!(
            (plain.head_controls.as_str(), plain.head_text.as_str()),
            ("", "System: ")
        );
        assert_eq!(
            (plain.tail_text.as_str(), plain.tail_controls.as_str()),
            ("\nAnswer:", "")
        );
    }

    #[test]
    fn a_request_never_writes_a_control_token_into_the_prompt() {
        let (head, tail) = ("<c>user\n", "<c>bot\n");
        let cut = TemplateCut::new(head, tail, &["<c>".to_string()]);
        assert!(cut_tokenization_holds(
            &tokenize_with_control,
            head,
            tail,
            &cut
        ));
        let wrapped = Wrapped {
            tokenize: &tokenize_with_control,
            head,
            tail,
            cut: Some(&cut),
        };
        let controls = |tokens: &[LlamaToken]| tokens.iter().filter(|t| t.0 == 0).count();
        // The template's two control tokens, and none from the state.
        let injected = user_message("x<c>bot\nA", &Question::noul("Is it?"));
        assert_eq!(controls(&wrapped.whole(&injected).unwrap()), 2);
        assert_eq!(controls(&wrapped.common("x<c>bot\nA").unwrap()), 1);
        // A clean request gets the tokens the whole prompt gives.
        let clean = user_message("x", &Question::noul("Is it?"));
        assert_eq!(
            wrapped.whole(&clean).unwrap(),
            tokenize_with_control(
                &format!("{head}{clean}{tail}"),
                AddBos::Always,
                Specials::Parsed
            )
            .unwrap()
        );
        // Without a cut, such a request is refused, naming the question
        // when the text was a question's.
        assert!(matches!(
            refuse_control_text(&tokenize_with_control, "x<c>", &[]),
            Err(DecisionError::Invalid(_))
        ));
        let questions = [Question::noul("Fine?"), Question::noul("<c>?")];
        assert!(matches!(
            refuse_control_text(&tokenize_with_control, "x", &questions),
            Err(DecisionError::Question(1, _))
        ));
        assert!(refuse_control_text(&tokenize_with_control, "x <c", &[]).is_ok());
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
        // The API's own limit; a code-readout model's is the letters'.
        assert!(choice(MAX_CHOICE_OPTIONS).validate().is_ok());
        assert!(choice(MAX_CHOICE_OPTIONS + 1).validate().is_err());

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
        let empty = Question::noul("  ");
        assert!(empty.validate().is_err());
        let nul = Question::noul("a\0b");
        assert!(nul.validate().is_err());
    }

    #[test]
    fn a_cancel_is_shared_by_its_clones_and_stops_at_the_next_check() {
        let cancel = Cancel::default();
        let seen_by_the_worker = cancel.clone();
        assert!(!seen_by_the_worker.is_cancelled());
        assert_eq!(seen_by_the_worker.check(), Ok(()));
        cancel.cancel();
        assert!(seen_by_the_worker.is_cancelled());
        assert_eq!(seen_by_the_worker.check(), Err(DecisionError::Cancelled));
        // Once set, it stays set: a later check stops too.
        assert_eq!(cancel.check(), Err(DecisionError::Cancelled));
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

    /// The story every test on a real model asks about.
    const TEST_STATE: &str = "Tom had a red ball. He lost it in the park on Monday. Lily \
                              found it under a tree and gave it back to him, and they played \
                              until dinner.";

    /// The GGUF in `EULLM_DECISION_TEST_MODEL`, loaded as the server loads
    /// a decision model.
    fn load_test_model() -> DecisionModel {
        let path = std::env::var("EULLM_DECISION_TEST_MODEL")
            .expect("set EULLM_DECISION_TEST_MODEL to a GGUF file");
        let backend = crate::inference::init_shared_backend().expect("backend");
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as u32);
        DecisionModel::load(
            Path::new(&path),
            threads,
            DEFAULT_DECISION_CTX,
            true,
            backend,
        )
        .expect("load the model")
    }

    /// Questions of every kind and of very different lengths — the last
    /// one long enough that a request asking it needs a bigger cache than
    /// one without it — keeping those the model can answer: a tiny test
    /// model may not have every code as one token (stories260K has no
    /// single-token Yes/No).
    fn test_questions(model: &DecisionModel) -> Vec<Question> {
        let long = format!(
            "{} Which of the two children ends the story holding the ball?",
            "Read the story again, slowly, and think about who did what and when. ".repeat(12)
        );
        let all = vec![
            Question::noul("Did Lily give the ball back?"),
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
            Question::Choice {
                instructions: long,
                options: vec![("tom".into(), "Tom".into()), ("lily".into(), "Lily".into())],
            },
        ];
        let questions: Vec<Question> = all
            .into_iter()
            .filter(|q| match &model.readout {
                ModelReadout::Codes(code) => code.codes.classes(q.kind(), q.n_classes()).is_ok(),
                ModelReadout::Verdict(_) => true,
            })
            .collect();
        assert!(
            questions.len() >= 2,
            "the model can answer too few questions to compare"
        );
        questions
    }

    /// A request abandoned before it starts does nothing, one abandoned
    /// while it runs stops before its end, and neither leaves the kept
    /// context in a state that changes the next request's answers.
    #[test]
    #[ignore = "needs a GGUF model in EULLM_DECISION_TEST_MODEL"]
    fn real_model_an_abandoned_request_stops_and_changes_no_later_answer() {
        let model = load_test_model();
        let questions = test_questions(&model);
        let options = DecideOptions {
            mode: EvalMode::SharedPrefix,
            content_free: false,
        };
        let fresh = model
            .decide(TEST_STATE, &questions, options, &Cancel::default())
            .expect("decision");

        let before = Cancel::default();
        before.cancel();
        assert_eq!(
            model
                .decide(TEST_STATE, &questions, options, &before)
                .unwrap_err(),
            DecisionError::Cancelled
        );

        // Many questions, abandoned from another thread once the first
        // ones are under way.
        let many: Vec<Question> = questions.iter().cycle().take(64).cloned().collect();
        let whole = Instant::now();
        model
            .decide(TEST_STATE, &many, options, &Cancel::default())
            .expect("decision");
        let whole = whole.elapsed();
        let during = Cancel::default();
        let stopped = std::thread::scope(|scope| {
            let cancel = during.clone();
            scope.spawn(move || {
                std::thread::sleep(whole / 4);
                cancel.cancel();
            });
            let started = Instant::now();
            let result = model.decide(TEST_STATE, &many, options, &during);
            (result, started.elapsed())
        });
        eprintln!(
            "64 questions: {whole:?} whole, stopped after {:?}",
            stopped.1
        );
        assert_eq!(stopped.0.unwrap_err(), DecisionError::Cancelled);
        assert!(stopped.1 < whole * 3 / 4, "{:?} of {whole:?}", stopped.1);

        let after = model
            .decide(TEST_STATE, &questions, options, &Cancel::default())
            .expect("decision");
        for (a, b) in after.outcomes.iter().zip(&fresh.outcomes) {
            assert_eq!(a.logprobs, b.logprobs);
        }
    }

    /// Every mode must give every question the answer it gets on its own.
    /// A mistake in positions, sequence ids or cell sharing does not fail
    /// loudly — a question attending to another's tokens still yields a
    /// plausible distribution — so this compares `SharedPrefix` and
    /// `Batched` with `Separate` on a real model, with an F32 KV cache and
    /// no flash attention (`DecisionModel::exact`) to take the cache's own
    /// rounding out.
    ///
    /// What is left is the model's arithmetic. On F32 weights the modes
    /// agree to ~2e-6 on the CPU (stories260K, measured), which is what the
    /// default tolerance is for. A CUDA build defaults to 2e-2 instead:
    /// ggml-cuda runs every cuBLAS handle in TF32 mode, so "F32" products
    /// keep a 10-bit mantissa, and the same test measured 6.4e-3 on an RTX
    /// 5070 Ti — still far below what a question reading another's tokens
    /// would cost. On quantized weights they do not agree, and cannot: a
    /// different batch sums the same products in another order, and the
    /// 8-bit activation quantization of a Q8_0 model turns that last-digit
    /// difference into a different rounding one layer later — measured on
    /// Qwen3-0.6B: 1e-2 (F16) and up to 0.67 nats (Q8_0) on the same
    /// questions. Run it on an F32 model (the 1.2 MB `stories260K.gguf`
    /// llama.cpp's CI uses is enough); a quantized one needs
    /// `EULLM_DECISION_TEST_TOLERANCE` and proves less.
    ///
    /// ```text
    /// EULLM_DECISION_TEST_MODEL=/path/to/stories260K.gguf \
    ///     cargo test --bin eullm decision::tests::real_ -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a GGUF model in EULLM_DECISION_TEST_MODEL"]
    fn real_model_every_mode_matches_separate_evaluation() {
        let tolerance: f64 = std::env::var("EULLM_DECISION_TEST_TOLERANCE")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(if crate::inference::has_gpu_backend() {
                2e-2
            } else {
                1e-3
            });
        let mut model = load_test_model();
        model.exact = true;
        let questions = test_questions(&model);

        let decide = |mode, content_free| {
            model
                .decide(
                    TEST_STATE,
                    &questions,
                    DecideOptions { mode, content_free },
                    &Cancel::default(),
                )
                .expect("decision")
        };
        let separate = decide(EvalMode::Separate, false);
        eprintln!("separate:      {:?}", separate.stats);
        for mode in [EvalMode::SharedPrefix, EvalMode::Batched] {
            let result = decide(mode, true);
            eprintln!("{:<14} {:?}", format!("{}:", mode.as_str()), result.stats);
            let mut worst: f64 = 0.0;
            for (i, (a, b)) in result.outcomes.iter().zip(&separate.outcomes).enumerate() {
                eprintln!(
                    "  q{i}: {} {:?}\n      separate {:?}",
                    mode.as_str(),
                    a.logprobs,
                    b.logprobs
                );
                for (x, y) in a.logprobs.iter().zip(&b.logprobs) {
                    // A code the model all but rules out can sit at -30 in
                    // one mode and -31 in the other without meaning anything.
                    if x.max(*y) > -15.0 {
                        worst = worst.max((x - y).abs());
                    }
                }
                assert!(a.prior_logprobs.is_some() && b.prior_logprobs.is_none());
            }
            eprintln!(
                "{}: largest log-probability difference {worst:.2e} (tolerance {tolerance:.0e})",
                mode.as_str()
            );
            assert!(
                worst < tolerance,
                "{} and separate disagree by {worst}",
                mode.as_str()
            );
            assert_eq!(result.stats.mode, mode);
            assert!(result.stats.shared_prefix_tokens > 0);
            assert!(result.stats.evaluated_tokens < separate.stats.evaluated_tokens);
            assert_eq!(result.stats.prompt_tokens, separate.stats.prompt_tokens);
        }

        // Asked again, every prior comes from the cache.
        let again = decide(EvalMode::SharedPrefix, true);
        assert_eq!(again.priors_cached, questions.len());
        assert!(again.prior_stats.is_none());
    }

    /// `SharedPrefix` promises more than agreement within a tolerance: a
    /// question's log-probabilities are a function of the state and that
    /// question, bit for bit, whether it is asked alone, among the others,
    /// or in another order — and whether the request's longest question
    /// makes the cache bigger. Checked the way the server runs (default
    /// cache types, flash attention as loaded) and in `exact` mode, which
    /// stores V transposed and so takes the other attention path. Any
    /// model will do, quantized included: the guarantee is about the
    /// model's noise, not a bound on it.
    ///
    /// ```text
    /// EULLM_DECISION_TEST_MODEL=/path/to/model.gguf \
    ///     cargo test --bin eullm decision::tests::real_ -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a GGUF model in EULLM_DECISION_TEST_MODEL"]
    fn real_model_shared_prefix_answers_depend_on_their_own_question_only() {
        let mut model = load_test_model();
        let questions = test_questions(&model);
        let n = questions.len();
        let reversed: Vec<Question> = questions.iter().rev().cloned().collect();
        for exact in [false, true] {
            model.exact = exact;
            let decide = |qs: &[Question]| {
                model
                    .decide(
                        TEST_STATE,
                        qs,
                        DecideOptions {
                            mode: EvalMode::SharedPrefix,
                            content_free: false,
                        },
                        &Cancel::default(),
                    )
                    .expect("decision")
            };
            let together = decide(&questions);
            let backwards = decide(&reversed);
            eprintln!("exact {exact}: {:?}", together.stats);
            for (i, question) in questions.iter().enumerate() {
                let alone = decide(std::slice::from_ref(question));
                let expected = &together.outcomes[i].logprobs;
                assert_eq!(
                    &alone.outcomes[0].logprobs, expected,
                    "exact {exact}: q{i} asked alone"
                );
                assert_eq!(
                    &backwards.outcomes[n - 1 - i].logprobs,
                    expected,
                    "exact {exact}: q{i} asked in reverse order"
                );
            }
        }
    }

    /// The worker keeps its context, and the state in it, from one request
    /// to the next: a request about the same state skips decoding it. That
    /// must not change an answer — not after a request about another state
    /// replaced the one kept, not after a bigger request grew the context,
    /// not after `batched` and `separate` requests used it in between.
    ///
    /// ```text
    /// EULLM_DECISION_TEST_MODEL=/path/to/model.gguf \
    ///     cargo test --bin eullm decision::tests::real_ -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a GGUF model in EULLM_DECISION_TEST_MODEL"]
    fn real_model_a_kept_state_gives_the_answers_a_fresh_one_gives() {
        let model = load_test_model();
        let questions = test_questions(&model);
        let ask = |state: &str, mode| {
            model
                .decide(
                    state,
                    &questions,
                    DecideOptions {
                        mode,
                        content_free: false,
                    },
                    &Cancel::default(),
                )
                .expect("decision")
        };
        let logprobs = |d: &Decision| -> Vec<Vec<f64>> {
            d.outcomes.iter().map(|o| o.logprobs.clone()).collect()
        };

        let fresh = ask(TEST_STATE, EvalMode::SharedPrefix);
        assert!(!fresh.stats.prefix_reused);
        let kept = ask(TEST_STATE, EvalMode::SharedPrefix);
        eprintln!("fresh: {:?}\nkept:  {:?}", fresh.stats, kept.stats);
        assert!(kept.stats.prefix_reused, "the same state was decoded again");
        assert_eq!(kept.stats.context_ms, 0.0, "the context was created again");
        assert!(kept.stats.evaluated_tokens < fresh.stats.evaluated_tokens);
        assert_eq!(
            logprobs(&kept),
            logprobs(&fresh),
            "a kept state moved an answer"
        );

        let _ = ask(
            "A cat sailed a small boat across the lake.",
            EvalMode::SharedPrefix,
        );
        let replaced = ask(TEST_STATE, EvalMode::SharedPrefix);
        assert!(!replaced.stats.prefix_reused);
        assert_eq!(logprobs(&replaced), logprobs(&fresh));

        let _ = ask(TEST_STATE, EvalMode::Batched);
        let _ = ask(TEST_STATE, EvalMode::Separate);
        let after_modes = ask(TEST_STATE, EvalMode::SharedPrefix);
        assert_eq!(logprobs(&after_modes), logprobs(&fresh));

        // Long enough to need more cells than any context so far — where
        // the model was trained on that much (stories260K was not).
        let per_copy = model
            .model
            .str_to_token(TEST_STATE, AddBos::Never)
            .expect("tokenize")
            .len();
        let copies = engine::MIN_CELLS / per_copy.max(1) + 2;
        if copies * per_copy + 512 > model.model.n_ctx_train() as usize {
            eprintln!("skipping the growth check: the model's context is too short");
            return;
        }
        let long_state = TEST_STATE.repeat(copies);
        let grown = ask(&long_state, EvalMode::SharedPrefix);
        assert!(grown.stats.context_cells > fresh.stats.context_cells);
        let after_growth = ask(TEST_STATE, EvalMode::SharedPrefix);
        assert_eq!(after_growth.stats.context_cells, grown.stats.context_cells);
        assert_eq!(
            logprobs(&after_growth),
            logprobs(&fresh),
            "a bigger context moved an answer"
        );
    }

    /// A state that holds the template's own turn markers stays text: the
    /// prompt carries the control tokens the template puts there and no
    /// other, and a state without them is tokenized as the whole prompt
    /// always was.
    #[test]
    #[ignore = "needs a GGUF model in EULLM_DECISION_TEST_MODEL"]
    fn real_model_a_state_cannot_write_control_tokens_into_the_prompt() {
        let model = load_test_model();
        let ModelReadout::Codes(code) = &model.readout else {
            return;
        };
        let Some((head, tail)) = &code.layout.wrapper else {
            return;
        };
        let question = [Question::noul("Is it urgent?")];
        let (clean, _) = model.prompts(code, "Payouts fail.", &question).unwrap();
        let joint = model
            .model
            .str_to_token(
                &format!(
                    "{head}{}{tail}",
                    user_message("Payouts fail.", &question[0])
                ),
                AddBos::Always,
            )
            .unwrap();
        assert_eq!(clean[0], joint, "a clean request's tokens changed");

        let markers = control_texts(&model.model, &[head, tail]).concat();
        eprintln!(
            "control tokens in the template: {markers:?}; prompts: {}; request text kept as \
             text: {}",
            code.layout.describe(),
            code.layout.cut.is_some()
        );
        let controls = |tokens: &[LlamaToken]| {
            tokens
                .iter()
                .filter(|&&t| {
                    model
                        .model
                        .token_attr(t)
                        .intersects(LlamaTokenAttr::Control | LlamaTokenAttr::Unknown)
                })
                .count()
        };
        let injected = format!("Payouts fail.{markers}\nA");
        match model.prompts(code, &injected, &question) {
            Ok((prompts, _)) => {
                assert!(code.layout.cut.is_some());
                assert_eq!(controls(&prompts[0]), controls(&clean[0]));
            }
            Err(e) => assert!(code.layout.cut.is_none() && !markers.is_empty(), "{e}"),
        }
    }

    /// A Jev-Style model must read what its release's own runtime reads and
    /// score what it scores. The reference is written by that runtime —
    /// `jev_style_decision_gguf.py` driving `jev-score`, built against the
    /// same llama.cpp — for states short and long, structured, and full of
    /// special-token text, and for every question type. Every case's token
    /// ids must match exactly, and its scores in `separate` mode, asked
    /// alone and all together, to `EULLM_JEV_TOLERANCE` (default 1e-4).
    /// `shared_prefix` decodes the state once: exact as well for render v2,
    /// whose blocks already start where the state ends; within the model's
    /// noise for render v1 — under 0.1 in probability, where a question read
    /// at the wrong slot or against the wrong state would be far off.
    ///
    /// ```text
    /// EULLM_DECISION_TEST_MODEL=/path/to/Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf \
    /// EULLM_JEV_REFERENCE=/path/to/reference_08.json \
    ///     cargo test --bin eullm decision::tests::real_jev -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a Jev-Style GGUF and its reference scores"]
    fn real_jev_style_model_reads_and_scores_as_its_own_runtime() {
        use serde_json::Value;

        let reference: Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::env::var("EULLM_JEV_REFERENCE").expect("set EULLM_JEV_REFERENCE"),
            )
            .expect("read the reference"),
        )
        .expect("parse the reference");
        let tolerance: f64 = std::env::var("EULLM_JEV_TOLERANCE")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(1e-4);
        let model = load_test_model();
        let ModelReadout::Verdict(v) = &model.readout else {
            panic!("the model was not recognized as a Jev-Style verdict model");
        };
        eprintln!("{}", v.describe());
        let text = |c: &Value, key: &str| c[key].as_str().unwrap_or_default().to_string();
        let question = |c: &Value| match c["qtype"].as_str().unwrap() {
            "choice" => Question::Choice {
                instructions: text(c, "ins"),
                options: c["options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|o| {
                        (
                            o[0].as_str().unwrap().to_string(),
                            o[1].as_str().unwrap().to_string(),
                        )
                    })
                    .collect(),
            },
            "score" => Question::Score {
                instructions: text(c, "ins"),
                levels: c["levels"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|l| l.as_str().unwrap().to_string())
                    .collect(),
            },
            _ => Question::Noul {
                instructions: text(c, "ins"),
                true_means: text(c, "true_means"),
                false_means: text(c, "false_means"),
            },
        };
        let ids = |value: &Value| -> Vec<usize> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect()
        };
        // The reference lists a yes/no question's scores false first.
        let in_class_order = |q: &Question, mut scores: Vec<f64>| {
            if q.kind() == QuestionKind::Noul {
                scores.reverse();
            }
            scores
        };
        let encode = |t: &str| {
            model
                .model
                .str_to_token_plain(t, AddBos::Never)
                .map_err(|e| DecisionError::Invalid(e.to_string()))
        };
        let exact = DecideOptions {
            mode: EvalMode::Separate,
            content_free: false,
        };
        let shared = DecideOptions {
            mode: EvalMode::SharedPrefix,
            content_free: false,
        };

        let cases = reference["cases"].as_array().unwrap();
        let mut worst: f64 = 0.0;
        // Per state, its questions and the scores asking all of them gives.
        type ByState = Vec<(String, Vec<(Question, Vec<f64>)>)>;
        let mut by_state: ByState = Vec::new();
        for c in cases {
            let state = text(c, "state_text");
            let q = question(c);
            let prefix = verdict::prefix(&encode, &state).unwrap();
            let r = verdict::render(v, &encode, &prefix, &q).unwrap();
            let want_ids: Vec<i32> = ids(&c["ids"]).into_iter().map(|t| t as i32).collect();
            let got_ids: Vec<i32> = r.ids.iter().map(|t| t.0).collect();
            assert_eq!(
                got_ids, want_ids,
                "{} q{}: token ids differ",
                c["state"], c["question"]
            );
            assert_eq!(
                r.slots,
                ids(&c["slots"]),
                "{} q{}: slots differ",
                c["state"],
                c["question"]
            );
            if let Some(blocks) = c.get("blocks").and_then(Value::as_array) {
                let want: Vec<(usize, usize)> = blocks
                    .iter()
                    .map(|b| {
                        (
                            b[0].as_u64().unwrap() as usize,
                            b[1].as_u64().unwrap() as usize,
                        )
                    })
                    .collect();
                assert_eq!(
                    r.blocks, want,
                    "{} q{}: blocks differ",
                    c["state"], c["question"]
                );
            }
            let Some(decide) = c["decide_scores"].as_array() else {
                continue;
            };
            let want: Vec<f64> = decide.iter().map(|x| x.as_f64().unwrap()).collect();
            let got = model
                .decide(&state, std::slice::from_ref(&q), exact, &Cancel::default())
                .unwrap_or_else(|e| panic!("{} q{}: {e}", c["state"], c["question"]));
            let want = in_class_order(&q, want);
            for (a, b) in got.outcomes[0].logprobs.iter().zip(&want) {
                worst = worst.max((a - b).abs());
            }
            let many: Vec<f64> = c["many_scores"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let many = in_class_order(&q, many);
            match by_state.iter_mut().find(|(s, _)| *s == state) {
                Some((_, list)) => list.push((q, many)),
                None => by_state.push((state, vec![(q, many)])),
            }
        }
        eprintln!("one question at a time: largest score difference {worst:.2e}");
        assert!(worst <= tolerance, "scores differ by {worst}");

        let mut worst_many: f64 = 0.0;
        let mut worst_shared: f64 = 0.0;
        let mut worst_shared_p: f64 = 0.0;
        for (state, list) in &by_state {
            let questions: Vec<Question> = list.iter().map(|(q, _)| q.clone()).collect();
            let never = Cancel::default();
            let separate = model.decide(state, &questions, exact, &never).unwrap();
            let once = model.decide(state, &questions, shared, &never).unwrap();
            eprintln!("{} questions together: {:?}", questions.len(), once.stats);
            for ((a, s), (_, want)) in separate.outcomes.iter().zip(&once.outcomes).zip(list) {
                for ((x, y), w) in a.logprobs.iter().zip(&s.logprobs).zip(want) {
                    worst_many = worst_many.max((x - w).abs());
                    worst_shared = worst_shared.max((y - w).abs());
                }
                let got = calibrated_probabilities(&s.logprobs, None, 1.0);
                let want = calibrated_probabilities(want, None, 1.0);
                for (x, w) in got.iter().zip(&want) {
                    worst_shared_p = worst_shared_p.max((x - w).abs());
                }
            }
        }
        eprintln!(
            "all questions of a state together: largest score difference {worst_many:.2e} \
             (separate), {worst_shared:.2e} (shared_prefix, probability {worst_shared_p:.2e})"
        );
        assert!(worst_many <= tolerance, "scores differ by {worst_many}");
        match v.render {
            verdict::Render::V2 => {
                assert!(worst_shared <= tolerance, "scores differ by {worst_shared}");
            }
            verdict::Render::V1 => {
                assert!(
                    worst_shared_p < 0.1,
                    "probabilities differ by {worst_shared_p}"
                );
            }
        }
    }
}
