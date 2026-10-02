//! `"model": "auto"`: which of the models configured with `--auto-model`
//! answers a request.
//!
//! Code filters first, and the model judges: a candidate that cannot read
//! the request's images or audio, or whose context is shorter than the
//! request, is not offered; among the others the decision model chooses,
//! with one `choice` question about a digest of the request
//! ([`routing_state`]), read from its logits as `/v1/systemone` reads any
//! question. The digest is built by code, not written by a model: a summary
//! would be a generation, with its latency, and a decision and its
//! arguments are separate things.
//!
//! The answer is the most likely option, with no threshold: a threshold is
//! only worth having once it has been calibrated on the domain's own data,
//! and the probabilities are kept everywhere — the response, the audit
//! trail — so that one can be fitted later. Whenever the decision model does
//! not decide — none is loaded, it did not answer within `--auto-timeout-ms`,
//! it failed — the fallback answers, and the reason says why.
//!
//! Every routing writes one `route` line to the audit trail. A decision that
//! finishes writes it from the thread it ran on, as `/v1/systemone` does, so
//! the record survives a client that left; one that times out is recorded
//! by the request instead, and the decision is cancelled — its answer was
//! never used. An atomic decides which side writes.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

use axum::http::{HeaderName, HeaderValue};
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

use super::systemone::{self, CancelOnDrop, OrderedMap};
use super::{AppState, KeepAlive, NamedModel};
use crate::audit::{
    AuditEntry, AuditLogger, DecisionRecord, ExcludedCandidate, RouteRef, RoutingRecord,
};
use crate::inference::decision::{
    self, Cancel, DecideOptions, DecisionError, DecisionModel, EvalMode, Question,
};

/// The model name a request routes with.
pub const AUTO: &str = "auto";

/// Fewest models `--auto-model` may name: one is not a choice.
pub const MIN_AUTO_MODELS: usize = 2;

/// Most models `--auto-model` may name. A decision model tells a few options
/// apart better than many, which is what the Reflex benchmarks measured; and
/// each option's description is read on every routed request.
pub const MAX_AUTO_MODELS: usize = 8;

/// Longest description an `--auto-model` may carry, in characters, so that
/// the question with all its options stays within the 2,048 tokens a
/// Jev-Style 0.8B reads for one question.
pub const MAX_DESCRIPTION_CHARS: usize = 400;

/// `--auto-timeout-ms` by default: the most routing may add to a request.
pub const DEFAULT_AUTO_TIMEOUT_MS: u64 = 1000;

/// The question the decision model answers, its options the candidates
/// offered, each with its description.
pub(crate) const ROUTE_QUESTION: &str = "Which model should answer the latest message? Choose \
the smallest model that will answer it correctly and completely; choose a larger one only when \
the message needs more reasoning, knowledge, code or length than a smaller one can give.";

/// Most characters of the digest the decision model reads: about 1,500
/// tokens, well within a decision model's context of 8,192.
const MAX_STATE_CHARS: usize = 6000;
/// How much of the system instructions and of each earlier turn it keeps.
const SNIPPET_CHARS: usize = 300;
/// The latest message is kept whole up to this length...
const LATEST_WHOLE_CHARS: usize = 4000;
/// ...and beyond it, its start and its end.
const LATEST_HEAD_CHARS: usize = 3000;
const LATEST_TAIL_CHARS: usize = 1000;
/// Most characters of tool names the digest lists.
const MAX_TOOL_NAMES_CHARS: usize = 400;
/// Characters per token, for the context filter's estimate of a request's
/// length: a conservative average for English and the European languages.
const CHARS_PER_TOKEN: f64 = 3.5;

/// Where a candidate's description came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptionSource {
    /// The text after `=` in `--auto-model NAME=DESCRIPTION`: written to
    /// say which requests the model should get.
    Flag,
    /// The description in the model's store manifest.
    Store,
    /// The catalog's description, with the model's size and domain: a
    /// product blurb ("Runs on any laptop CPU"), not which requests it
    /// should get.
    Catalog,
    /// None was found: the option is the model's name alone.
    Name,
}

impl DescriptionSource {
    /// Where it came from, as the startup log says it.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Flag => "--auto-model",
            Self::Store => "the model store",
            Self::Catalog => {
                "the catalog, a product description: say which requests it should get with \
                 --auto-model NAME=DESCRIPTION"
            }
            Self::Name => {
                "none found, the name alone: say which requests it should get with --auto-model \
                 NAME=DESCRIPTION"
            }
        }
    }

    /// Whether the startup log warns about it.
    pub fn warns(self) -> bool {
        matches!(self, Self::Catalog | Self::Name)
    }
}

/// One model `"model": "auto"` may choose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteCandidate {
    /// The name requests use for it, and the option's name.
    pub name: String,
    /// Its GGUF.
    pub path: PathBuf,
    /// What the decision model reads about it.
    pub description: String,
    pub source: DescriptionSource,
    /// It can read images and audio: a projector is in the store beside it,
    /// or `--mmproj` gives every model one.
    pub has_projector: bool,
}

/// The models `"model": "auto"` chooses between, from `--auto-model`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteTable {
    /// In the order configured, which is the order the decision model reads
    /// them in: smallest first.
    pub candidates: Vec<RouteCandidate>,
    /// The candidate that answers whenever the decision model does not
    /// decide: `--default-model` when it is one of them, the last otherwise.
    pub fallback: usize,
    /// `--auto-timeout-ms`.
    pub timeout: Duration,
}

/// What resolving a model named by `--auto-model` finds out about it.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateFacts {
    pub model: NamedModel,
    /// What its store manifest says it is, when someone wrote that: not an
    /// external pull's provenance, nor the catalog's text copied there.
    pub store_description: Option<String>,
    pub catalog: Option<CatalogFacts>,
    pub has_projector: bool,
}

/// What the catalog says about a candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogFacts {
    pub description: String,
    pub params_b: f32,
    pub domain: String,
}

impl RouteTable {
    /// The table `--auto-model` describes, `None` without one. Each spec is
    /// `NAME` or `NAME=DESCRIPTION`, split at the first `=`; `lookup`
    /// resolves a name as the command line resolves `--decision-model`.
    /// Refused: fewer than two models or more than eight, `auto` itself, a
    /// model given twice (by its name or as the same file), a name that is no
    /// model, and a description over 400 characters or holding a NUL.
    pub fn resolve(
        specs: &[String],
        default_model: Option<&NamedModel>,
        timeout: Duration,
        lookup: impl Fn(&str) -> Option<CandidateFacts>,
    ) -> Result<Option<Self>, String> {
        if specs.is_empty() {
            return Ok(None);
        }
        if !(MIN_AUTO_MODELS..=MAX_AUTO_MODELS).contains(&specs.len()) {
            return Err(format!(
                "--auto-model takes {MIN_AUTO_MODELS} to {MAX_AUTO_MODELS} models to choose \
                 between, got {}",
                specs.len()
            ));
        }
        let mut candidates: Vec<RouteCandidate> = Vec::with_capacity(specs.len());
        for spec in specs {
            let (name, text) = match spec.split_once('=') {
                Some((name, text)) => (name.trim(), Some(text.trim())),
                None => (spec.trim(), None),
            };
            if name.is_empty() {
                return Err(format!(
                    "--auto-model '{spec}': a model name comes before '='"
                ));
            }
            if name.eq_ignore_ascii_case(AUTO) {
                return Err(format!(
                    "--auto-model '{name}': `{AUTO}` is the router, not a model it can choose"
                ));
            }
            if let Some(text) = text {
                if text.chars().count() > MAX_DESCRIPTION_CHARS {
                    return Err(format!(
                        "--auto-model '{name}': the description is {} characters, and at most \
                         {MAX_DESCRIPTION_CHARS} keep the question within what a decision \
                         model reads",
                        text.chars().count()
                    ));
                }
                if text.contains('\0') {
                    return Err(format!(
                        "--auto-model '{name}': the description holds a NUL character"
                    ));
                }
            }
            let facts = lookup(name).ok_or_else(|| {
                format!(
                    "--auto-model '{name}' is not a model: give a GGUF path or a name `eullm \
                     list` shows (a catalog model has to be pulled first: eullm pull {name})"
                )
            })?;
            if let Some(twin) = candidates.iter().find(|c| same_model(c, &facts.model)) {
                return Err(format!(
                    "--auto-model names {} twice ('{}' and '{name}')",
                    facts.model.name, twin.name
                ));
            }
            let (description, source) = describe(&facts, text);
            candidates.push(RouteCandidate {
                name: facts.model.name,
                path: facts.model.path,
                description,
                source,
                has_projector: facts.has_projector,
            });
        }
        let fallback = default_model
            .and_then(|default| candidates.iter().position(|c| same_model(c, default)))
            .unwrap_or(candidates.len() - 1);
        Ok(Some(Self {
            candidates,
            fallback,
            timeout,
        }))
    }

    /// The candidate that answers when the decision model does not decide.
    pub fn fallback(&self) -> &RouteCandidate {
        &self.candidates[self.fallback]
    }
}

/// Whether a candidate is `model`: the same name, as names are compared
/// everywhere (`model_names_match`), or the same file under another name.
fn same_model(candidate: &RouteCandidate, model: &NamedModel) -> bool {
    super::model_names_match(&candidate.name, &model.name)
        || super::same_file(&candidate.path, &model.path)
}

/// A candidate's description and where it came from, in order of
/// preference: the flag's, the store's, the catalog's with the model's size
/// and domain, the name.
fn describe(facts: &CandidateFacts, flag: Option<&str>) -> (String, DescriptionSource) {
    if let Some(text) = flag.filter(|t| !t.is_empty()) {
        return (text.to_string(), DescriptionSource::Flag);
    }
    if let Some(text) = &facts.store_description {
        return (bounded(text), DescriptionSource::Store);
    }
    if let Some(catalog) = &facts.catalog {
        let text = format!(
            "{} (about {} billion parameters, domain {})",
            catalog.description.trim(),
            catalog.params_b,
            catalog.domain
        );
        return (bounded(&text), DescriptionSource::Catalog);
    }
    (facts.model.name.clone(), DescriptionSource::Name)
}

/// A description nobody wrote for this purpose, made one the decision model
/// can read: no NUL, and cut to [`MAX_DESCRIPTION_CHARS`].
fn bounded(text: &str) -> String {
    let clean: String = text.chars().filter(|&c| c != '\0').collect();
    start_of(&clean, MAX_DESCRIPTION_CHARS)
}

// ── The request, as the router reads it ─────────────────────────────────

/// A request as the router reads it.
pub(crate) enum RouteInput<'a> {
    /// `/api/chat` and `/v1/chat/completions`: the conversation, and the
    /// names of the tools it offers.
    Chat {
        messages: &'a [Value],
        tools: Vec<String>,
    },
    /// `/api/generate`: the prompt.
    Generate { prompt: &'a str },
}

impl<'a> RouteInput<'a> {
    /// The input of a request body of any of the three endpoints:
    /// `messages` makes it a chat, else `prompt` a generation.
    pub(crate) fn of(body: &'a Value) -> Option<Self> {
        if let Some(messages) = body.get("messages").and_then(Value::as_array) {
            let tools = body
                .get("tools")
                .and_then(Value::as_array)
                .map(|tools| {
                    tools
                        .iter()
                        .filter_map(|t| {
                            t.pointer("/function/name")
                                .or_else(|| t.get("name"))
                                .and_then(Value::as_str)
                        })
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            return Some(Self::Chat { messages, tools });
        }
        let prompt = body.get("prompt").and_then(Value::as_str)?;
        Some(Self::Generate { prompt })
    }
}

/// What the code filters read about a request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RequestFacts {
    /// Images and audio clips it carries.
    pub(crate) attachments: usize,
    /// Characters of text it carries, every message's.
    pub(crate) chars: usize,
}

impl RequestFacts {
    /// About how many tokens the request takes.
    pub(crate) fn estimated_tokens(&self) -> u32 {
        (self.chars as f64 / CHARS_PER_TOKEN)
            .ceil()
            .min(f64::from(u32::MAX)) as u32
    }

    pub(crate) fn of(input: &RouteInput<'_>) -> Self {
        match input {
            RouteInput::Generate { prompt } => Self {
                attachments: 0,
                chars: prompt.chars().count(),
            },
            RouteInput::Chat { messages, .. } => messages.iter().map(message_text).fold(
                Self::default(),
                |facts, (_, text, attachments)| Self {
                    attachments: facts.attachments + attachments,
                    chars: facts.chars + text.chars().count(),
                },
            ),
        }
    }
}

/// A message's role, its text and how many attachments it carries: a string
/// content as it is; OpenAI's array of parts reduced to its text parts, the
/// image and audio parts counted; Ollama's `images` counted.
fn message_text(message: &Value) -> (&str, String, usize) {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("user");
    let mut attachments = message
        .get("images")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let text = match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => {
            let mut texts = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") | Some("input_text") | None => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            texts.push(text);
                        }
                    }
                    Some("image_url" | "input_image" | "image" | "input_audio" | "audio") => {
                        attachments += 1;
                    }
                    Some(_) => {}
                }
            }
            texts.join("\n")
        }
        _ => String::new(),
    };
    (role, text, attachments)
}

/// The text without NUL characters, which no decision model reads.
fn clean(text: &str) -> String {
    text.chars().filter(|&c| c != '\0').collect()
}

/// The text on one line: every run of whitespace a single space.
fn collapsed(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The first `n` characters.
fn start_of(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

/// The last `n` characters.
fn end_of(text: &str, n: usize) -> String {
    let count = text.chars().count();
    text.chars().skip(count.saturating_sub(n)).collect()
}

/// A message too long to read whole, as its start and its end: what it asks
/// is usually at one or the other.
fn head_and_tail(text: &str) -> String {
    let count = text.chars().count();
    if count <= LATEST_WHOLE_CHARS {
        return text.to_string();
    }
    let left_out = count - LATEST_HEAD_CHARS - LATEST_TAIL_CHARS;
    format!(
        "{}\n[… {left_out} characters left out …]\n{}",
        start_of(text, LATEST_HEAD_CHARS),
        end_of(text, LATEST_TAIL_CHARS)
    )
}

/// The request as the decision model reads it: a digest built by code, the
/// same for the same request, and bounded — at most [`MAX_STATE_CHARS`]
/// characters whatever the request holds.
///
/// For a conversation, what it is (how many messages, attachments, tools),
/// the start of the system instructions, the two turns before the latest
/// message — a follow-up such as "and in Python?" cannot be judged without
/// them — and the latest message, whole up to 4,000 characters, its start
/// and end beyond. For a prompt, the prompt, the same way.
pub(crate) fn routing_state(input: &RouteInput<'_>) -> String {
    let mut state = String::from("Request to answer, with its context.\n");
    match input {
        RouteInput::Generate { prompt } => {
            state.push_str("Prompt:\n");
            state.push_str(&head_and_tail(&clean(prompt)));
        }
        RouteInput::Chat { messages, tools } => {
            let turns: Vec<(&str, String, usize)> = messages.iter().map(message_text).collect();
            let users = turns.iter().filter(|(role, ..)| *role == "user").count();
            let attachments: usize = turns.iter().map(|(.., n)| n).sum();
            state.push_str(&format!(
                "Conversation: {} message{}, {users} from the user. Attachments: {}. Tools \
                 offered: {}.\n",
                turns.len(),
                if turns.len() == 1 { "" } else { "s" },
                if attachments == 0 {
                    "none".to_string()
                } else {
                    attachments.to_string()
                },
                tool_names(tools),
            ));
            if let Some((_, system, _)) = turns
                .iter()
                .find(|(role, text, _)| *role == "system" && !text.trim().is_empty())
            {
                state.push_str(&format!(
                    "System instructions (start): {}\n",
                    start_of(&collapsed(&clean(system)), SNIPPET_CHARS)
                ));
            }
            let latest = turns
                .iter()
                .rposition(|(role, ..)| *role == "user")
                .unwrap_or(turns.len().saturating_sub(1));
            let earlier: Vec<String> = turns[..latest.min(turns.len())]
                .iter()
                .rev()
                .filter_map(|(role, text, _)| {
                    let text = collapsed(&clean(text));
                    match *role {
                        "user" => Some(format!("user (start): {}", start_of(&text, SNIPPET_CHARS))),
                        "assistant" => {
                            Some(format!("assistant (end): {}", end_of(&text, SNIPPET_CHARS)))
                        }
                        _ => None,
                    }
                })
                .take(2)
                .collect();
            if !earlier.is_empty() {
                state.push_str("Earlier turns (most recent last):\n");
                for turn in earlier.iter().rev() {
                    state.push_str(turn);
                    state.push('\n');
                }
            }
            state.push_str("Latest message:\n");
            if let Some((_, text, _)) = turns.get(latest) {
                state.push_str(&head_and_tail(&clean(text)));
            }
        }
    }
    if state.chars().count() > MAX_STATE_CHARS {
        state = start_of(&state, MAX_STATE_CHARS);
    }
    state
}

/// The tools a request offers, by name, as far as fits in
/// [`MAX_TOOL_NAMES_CHARS`]; "none" for none.
fn tool_names(tools: &[String]) -> String {
    if tools.is_empty() {
        return "none".to_string();
    }
    let mut listed = String::new();
    for (i, name) in tools.iter().enumerate() {
        let name = collapsed(&clean(name));
        if listed.chars().count() + name.chars().count() + 2 > MAX_TOOL_NAMES_CHARS {
            let rest = tools.len() - i;
            return if listed.is_empty() {
                format!("{rest} tools")
            } else {
                format!("{listed} and {rest} more")
            };
        }
        if !listed.is_empty() {
            listed.push_str(", ");
        }
        listed.push_str(&name);
    }
    listed
}

// ── Which candidates are offered ────────────────────────────────────────

/// The candidates offered for a request, as indices into the table, and
/// the others with why: one that cannot read the request's attachments, or
/// whose context per request (`contexts`, in the table's order) is shorter
/// than the request is estimated to be (characters / 3.5).
pub(crate) fn eligible(
    table: &RouteTable,
    facts: RequestFacts,
    contexts: &[u32],
) -> (Vec<usize>, Vec<ExcludedCandidate>) {
    let needed = facts.estimated_tokens();
    let mut offered = Vec::new();
    let mut excluded = Vec::new();
    for (i, candidate) in table.candidates.iter().enumerate() {
        let context = contexts.get(i).copied().unwrap_or(u32::MAX);
        let why = if facts.attachments > 0 && !candidate.has_projector {
            Some(format!(
                "it cannot read images or audio, and the request carries {}",
                facts.attachments
            ))
        } else if needed > context {
            Some(format!(
                "its context of {context} tokens per request is shorter than the request, \
                 about {needed} tokens"
            ))
        } else {
            None
        };
        match why {
            Some(why) => excluded.push(ExcludedCandidate {
                model: candidate.name.clone(),
                why,
            }),
            None => offered.push(i),
        }
    }
    (offered, excluded)
}

/// The question the decision model answers: [`ROUTE_QUESTION`], the offered
/// candidates its options, in the table's order.
pub(crate) fn question(table: &RouteTable, offered: &[usize]) -> Question {
    Question::Choice {
        instructions: ROUTE_QUESTION.to_string(),
        options: offered
            .iter()
            .map(|&i| {
                let candidate = &table.candidates[i];
                (candidate.name.clone(), candidate.description.clone())
            })
            .collect(),
    }
}

// ── Deciding ────────────────────────────────────────────────────────────

/// Why a request went to the model it went to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteReason {
    /// The decision model chose it.
    Decided,
    /// No decision model is loaded: no `--decision-model`, or it was
    /// unloaded.
    NoDecisionModel,
    /// The decision model did not answer within `--auto-timeout-ms`.
    Timeout,
    /// The decision model could not decide: input over its budget, text it
    /// refuses, a failure in llama.cpp.
    DecisionError,
    /// No candidate can take the request; the fallback answers, and fails
    /// as a request to it would.
    NoEligibleCandidate,
    /// One candidate can take the request: there was nothing to decide.
    OnlyCandidate,
    /// The model chosen could not be loaded, and the fallback answered in
    /// its place. Known only once the route is taken: the route's audit
    /// line has the reason it was chosen for, and the generation's line
    /// says it fell back.
    LoadFailed,
}

impl RouteReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Decided => "decided",
            Self::NoDecisionModel => "no_decision_model",
            Self::Timeout => "timeout",
            Self::DecisionError => "decision_error",
            Self::NoEligibleCandidate => "no_eligible_candidate",
            Self::OnlyCandidate => "only_candidate",
            Self::LoadFailed => "load_failed",
        }
    }
}

/// Which model a request goes to, and everything the response and the
/// audit trail say about how.
#[derive(Debug, Clone)]
pub(crate) struct Route {
    /// The route's audit line's id, which names it everywhere.
    pub(crate) id: Uuid,
    /// The model that answers.
    pub(crate) model: String,
    pub(crate) reason: RouteReason,
    pub(crate) fallback: String,
    /// The candidates offered, as indices into the table.
    pub(crate) offered: Vec<usize>,
    /// Per candidate in the table, whether it was resident when routing
    /// began.
    pub(crate) resident: Vec<bool>,
    /// The decision model's probability for each offered candidate, when it
    /// decided.
    pub(crate) probabilities: Option<Vec<f64>>,
    pub(crate) confidence: Option<f64>,
    pub(crate) excluded: Vec<ExcludedCandidate>,
    pub(crate) decision_model: Option<String>,
    /// How long routing took, the decision included.
    pub(crate) decision_ms: f64,
    /// The digest the decision model read (or would have).
    pub(crate) state: String,
    pub(crate) question: Question,
    pub(crate) error: Option<String>,
}

/// Who writes a route's audit line: the decision when it finishes in time,
/// the request when it does not.
const PENDING: u8 = 0;
const DECIDED: u8 = 1;
const TIMED_OUT: u8 = 2;

/// What a route's audit line records besides its outcome.
#[derive(Debug, Clone)]
struct RouteLine {
    id: Uuid,
    user_id: Option<String>,
    fallback: String,
    candidates: Vec<String>,
    excluded: Vec<ExcludedCandidate>,
    dry_run: bool,
}

impl RouteLine {
    /// The line, with the decision model's record when it decided.
    fn entry(
        &self,
        model: &str,
        reason: RouteReason,
        decision_model: Option<&str>,
        decision_ms: f64,
        decision: Option<(DecisionRecord, usize)>,
        error: Option<String>,
    ) -> AuditEntry {
        let mut entry = AuditEntry::new(decision_model.unwrap_or(AUTO).to_string(), "route".into());
        entry.id = self.id;
        entry.duration_ms = decision_ms.round() as u64;
        entry.user_id = self.user_id.clone();
        if let Some((record, evaluated_tokens)) = decision {
            entry.input_tokens = u32::try_from(evaluated_tokens).unwrap_or(u32::MAX);
            entry.decision = Some(record);
        }
        entry.routing = Some(RoutingRecord {
            requested: AUTO.to_string(),
            model: model.to_string(),
            reason: reason.as_str().to_string(),
            fallback: self.fallback.clone(),
            candidates: self.candidates.clone(),
            excluded: self.excluded.clone(),
            decision_model: decision_model.map(str::to_string),
            decision_ms,
            dry_run: self.dry_run,
            error,
        });
        entry
    }
}

/// Route a request: filter the candidates, ask the decision model among the
/// rest within the table's timeout, and record the route. Never loads a
/// model: which one answers is the caller's to load. `contexts` and
/// `resident` describe each candidate in the table's order (see
/// `AppState::route_candidates`); `user_id` is who asked, for the audit
/// line; `dry_run` marks a route nothing is generated for.
pub(crate) async fn decide_route(
    state: &AppState,
    table: &RouteTable,
    input: &RouteInput<'_>,
    contexts: &[u32],
    resident: Vec<bool>,
    user_id: Option<String>,
    dry_run: bool,
) -> Route {
    let started = Instant::now();
    let (offered, excluded) = eligible(table, RequestFacts::of(input), contexts);
    let fallback = table.fallback().name.clone();
    let mut route = Route {
        id: Uuid::new_v4(),
        model: fallback.clone(),
        reason: RouteReason::NoEligibleCandidate,
        fallback,
        question: question(table, &offered),
        offered,
        resident,
        probabilities: None,
        confidence: None,
        excluded,
        decision_model: None,
        decision_ms: 0.0,
        state: routing_state(input),
        error: None,
    };
    let line = RouteLine {
        id: route.id,
        user_id,
        fallback: route.fallback.clone(),
        candidates: route
            .offered
            .iter()
            .map(|&i| table.candidates[i].name.clone())
            .collect(),
        excluded: route.excluded.clone(),
        dry_run,
    };
    let decided = match route.offered.as_slice() {
        [] => false,
        [only] => {
            route.reason = RouteReason::OnlyCandidate;
            route.model = table.candidates[*only].name.clone();
            false
        }
        _ => {
            let model = state
                .decision
                .read()
                .await
                .as_ref()
                .map(|slot| (slot.model_name.clone(), Arc::clone(&slot.model)));
            match model {
                None => {
                    route.reason = RouteReason::NoDecisionModel;
                    false
                }
                Some((name, model)) => {
                    // Decision traffic like any other: it keeps the decision
                    // model from expiring.
                    state.touch_decision_slot(KeepAlive::Default).await;
                    route.decision_model = Some(name.clone());
                    ask(&mut route, table, model, name, &line, started).await
                }
            }
        }
    };
    route.decision_ms = elapsed_ms(started);
    if !decided {
        AuditLogger::new().log(&line.entry(
            &route.model,
            route.reason,
            route.decision_model.as_deref(),
            route.decision_ms,
            None,
            route.error.clone(),
        ));
    }
    route
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

/// Ask the decision model, within the table's timeout, and fill `route`
/// with what came of it. Returns whether the decision's own thread wrote
/// the audit line; otherwise the caller writes it.
async fn ask(
    route: &mut Route,
    table: &RouteTable,
    model: Arc<DecisionModel>,
    model_name: String,
    line: &RouteLine,
    started: Instant,
) -> bool {
    let cancel = Cancel::default();
    // A request whose client is gone stops its decision at the next point
    // where it can, as `/v1/systemone` does.
    let _cancel_when_abandoned = CancelOnDrop(cancel.clone());
    let writer = Arc::new(AtomicU8::new(PENDING));
    let job = RouteJob {
        model,
        model_name,
        state: route.state.clone(),
        question: route.question.clone(),
        offered: line.candidates.clone(),
        cancel: cancel.clone(),
        writer: Arc::clone(&writer),
        line: line.clone(),
        started,
    };
    let mut running = tokio::task::spawn_blocking(move || job.run());
    let outcome = match tokio::time::timeout(table.timeout, &mut running).await {
        Ok(outcome) => outcome,
        Err(_) => {
            if writer
                .compare_exchange(PENDING, TIMED_OUT, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                cancel.cancel();
                route.reason = RouteReason::Timeout;
                return false;
            }
            // Decided, and recorded, as the time ran out: its answer is
            // moments away and it is the one on record.
            running.await
        }
    };
    match outcome {
        Ok(Ok(decided)) => {
            route.reason = RouteReason::Decided;
            route.model = line.candidates[decided.best].clone();
            route.confidence = Some(decided.confidence);
            route.probabilities = Some(decided.probabilities);
            true
        }
        Ok(Err(e)) => {
            tracing::warn!("Routing decision failed, the fallback answers: {e}");
            route.reason = RouteReason::DecisionError;
            route.error = Some(e.to_string());
            false
        }
        Err(e) => {
            tracing::warn!("Routing decision task failed, the fallback answers: {e}");
            route.reason = RouteReason::DecisionError;
            route.error = Some(format!("decision task failed: {e}"));
            false
        }
    }
}

/// One routing decision, run on a blocking thread: everything it and its
/// audit line need, owned.
struct RouteJob {
    model: Arc<DecisionModel>,
    model_name: String,
    state: String,
    question: Question,
    /// The offered candidates' names, in the question's order.
    offered: Vec<String>,
    cancel: Cancel,
    writer: Arc<AtomicU8>,
    line: RouteLine,
    started: Instant,
}

/// What a finished decision chose, and how sure it was.
struct Decided {
    probabilities: Vec<f64>,
    best: usize,
    confidence: f64,
}

impl RouteJob {
    /// Decide, then — unless the request gave up waiting first — write the
    /// route's audit line, here, so that it is written whatever became of
    /// the request.
    fn run(self) -> Result<Decided, DecisionError> {
        let readout = self.model.readout();
        let temperature = self.model.default_temperature();
        let options = DecideOptions {
            mode: EvalMode::SharedPrefix,
            content_free: false,
        };
        let decision = self.model.decide(
            &self.state,
            std::slice::from_ref(&self.question),
            options,
            &self.cancel,
        )?;
        let record = systemone::answer_record(
            "route",
            &self.question,
            &decision.outcomes[0],
            None,
            readout,
            temperature,
        );
        let best = systemone::most_likely(&record.probabilities);
        let decided = Decided {
            confidence: decision::max_probability_confidence(&record.probabilities),
            probabilities: record.probabilities.clone(),
            best,
        };
        if self
            .writer
            .compare_exchange(PENDING, DECIDED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let decision_ms = elapsed_ms(self.started);
            let record = DecisionRecord {
                state_sha256: systemone::sha256_hex(&self.state),
                readout: readout.as_str().to_string(),
                mode: decision.stats.mode.as_str().to_string(),
                calibration: "none".to_string(),
                temperature,
                confidence_method: decision::CONFIDENCE_METHOD.to_string(),
                client_disconnected: self.cancel.is_cancelled(),
                policy_removed: Default::default(),
                answers: vec![record],
            };
            AuditLogger::new().log(&self.line.entry(
                &self.offered[best],
                RouteReason::Decided,
                Some(&self.model_name),
                decision_ms,
                Some((record, decision.stats.evaluated_tokens)),
                None,
            ));
        }
        Ok(decided)
    }
}

// ── What `POST /api/route` answers ──────────────────────────────────────

/// `POST /api/route`'s answer: the route, and what the decision model read
/// — the state and the question, in `/v1/systemone`'s shape, so that a
/// benchmark can ask the same question, worded otherwise, about the very
/// state the server built.
#[derive(Debug, Serialize)]
pub(crate) struct RouteResponse {
    model: String,
    reason: &'static str,
    fallback: String,
    candidates: Vec<OfferedCandidate>,
    excluded: Vec<ExcludedCandidate>,
    confidence: Option<f64>,
    decision_model: Option<String>,
    decision_ms: f64,
    state: String,
    question: QuestionEcho,
    route_id: Uuid,
}

/// An offered candidate in [`RouteResponse`].
#[derive(Debug, Serialize)]
struct OfferedCandidate {
    model: String,
    description: String,
    /// The decision model's probability for it; `null` when it did not
    /// decide.
    probability: Option<f64>,
    /// It was loaded when the request was routed.
    resident: bool,
}

/// A question in `/v1/systemone`'s request shape, its options in order.
#[derive(Debug, Serialize)]
struct QuestionEcho {
    #[serde(rename = "type")]
    kind: &'static str,
    instructions: String,
    criteria: OrderedMap<String>,
}

impl RouteResponse {
    pub(crate) fn new(route: Route, table: &RouteTable) -> Self {
        let candidates = route
            .offered
            .iter()
            .enumerate()
            .map(|(n, &i)| {
                let candidate = &table.candidates[i];
                OfferedCandidate {
                    model: candidate.name.clone(),
                    description: candidate.description.clone(),
                    probability: route.probabilities.as_ref().and_then(|p| p.get(n).copied()),
                    resident: route.resident.get(i).copied().unwrap_or(false),
                }
            })
            .collect();
        let (instructions, options) = match &route.question {
            Question::Choice {
                instructions,
                options,
            } => (instructions.clone(), options.clone()),
            other => (other.instructions().to_string(), Vec::new()),
        };
        Self {
            model: route.model,
            reason: route.reason.as_str(),
            fallback: route.fallback,
            candidates,
            excluded: route.excluded,
            confidence: route.confidence,
            decision_model: route.decision_model,
            decision_ms: route.decision_ms,
            state: route.state,
            question: QuestionEcho {
                kind: "choice",
                instructions,
                criteria: OrderedMap(options),
            },
            route_id: route.id,
        }
    }
}

// ── What a routed request's response says ───────────────────────────────

/// The headers a routed response carries, before its body: the model that
/// answers, the reason, and the route's id.
pub(crate) const MODEL_HEADER: &str = "x-eullm-model";
pub(crate) const ROUTE_HEADER: &str = "x-eullm-route";
pub(crate) const ROUTE_ID_HEADER: &str = "x-eullm-route-id";

/// What the response to a request `"model": "auto"` routed says about its
/// route — the `eullm.route` object on the response, or on the last line or
/// chunk of a stream, and the `X-EuLLM-*` headers — and what the
/// generation's audit line links to. Ollama and OpenAI clients ignore an
/// object they do not know.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct RouteInfo {
    /// What the request named: `auto`.
    requested: &'static str,
    /// The model that answers.
    model: String,
    reason: &'static str,
    confidence: Option<f64>,
    /// Per offered candidate, in the order the decision model read them,
    /// the probability it gave it; `null` when it did not decide.
    probabilities: Option<OrderedMap<f64>>,
    decision_model: Option<String>,
    decision_ms: f64,
    /// The id of the route's audit line.
    id: Uuid,
    /// Why the model chosen did not answer, when the fallback did instead.
    #[serde(skip)]
    load_failed: Option<String>,
}

impl RouteInfo {
    /// What `route` says, its candidates named from `table`.
    pub(crate) fn new(route: &Route, table: &RouteTable) -> Self {
        let probabilities = route.probabilities.as_ref().map(|p| {
            OrderedMap(
                route
                    .offered
                    .iter()
                    .zip(p)
                    .map(|(&i, &p)| (table.candidates[i].name.clone(), p))
                    .collect(),
            )
        });
        Self {
            requested: AUTO,
            model: route.model.clone(),
            reason: route.reason.as_str(),
            confidence: route.confidence,
            probabilities,
            decision_model: route.decision_model.clone(),
            decision_ms: route.decision_ms,
            id: route.id,
            load_failed: None,
        }
    }

    /// The model chosen could not be loaded, for `why`: `fallback` answers.
    pub(crate) fn fell_back(&mut self, fallback: &str, why: String) {
        self.model = fallback.to_string();
        self.reason = RouteReason::LoadFailed.as_str();
        self.load_failed = Some(why);
    }

    /// What the generation's audit line records of the route.
    pub(crate) fn reference(&self) -> RouteRef {
        RouteRef {
            id: self.id,
            requested: self.requested.to_string(),
            fallback: self
                .load_failed
                .as_ref()
                .map(|why| format!("{}: {why}", RouteReason::LoadFailed.as_str())),
        }
    }

    /// The response's `X-EuLLM-*` headers. A value a header cannot carry —
    /// a model name with a control character in it — is left out rather
    /// than failing the response.
    pub(crate) fn headers(&self) -> Vec<(HeaderName, HeaderValue)> {
        [
            (MODEL_HEADER, self.model.clone()),
            (ROUTE_HEADER, self.reason.to_string()),
            (ROUTE_ID_HEADER, self.id.to_string()),
        ]
        .into_iter()
        .filter_map(|(name, value)| {
            let value = HeaderValue::from_str(&value).ok()?;
            Some((HeaderName::from_static(name), value))
        })
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn named(name: &str) -> NamedModel {
        NamedModel {
            name: name.to_string(),
            path: PathBuf::from(format!("/store/{name}/model.gguf")),
        }
    }

    /// A store of `qwen3-4b` (described in its manifest), `qwen3-8b` (in the
    /// catalog only), `custom` (described nowhere) and `vision` (with a
    /// projector), an Ollama tag of each standing for it as on the command
    /// line.
    fn lookup(name: &str) -> Option<CandidateFacts> {
        let name = name.replace(':', "-");
        let facts = |store: Option<&str>, catalog: bool, projector: bool| CandidateFacts {
            model: named(&name),
            store_description: store.map(str::to_string),
            catalog: catalog.then(|| CatalogFacts {
                description: "Mid-size Qwen3.".to_string(),
                params_b: 8.0,
                domain: "general".to_string(),
            }),
            has_projector: projector,
        };
        match name.as_str() {
            "qwen3-4b" => Some(facts(Some("Everyday chat."), true, false)),
            "qwen3-8b" => Some(facts(None, true, false)),
            "custom" => Some(facts(None, false, false)),
            "vision" => Some(facts(None, false, true)),
            _ => None,
        }
    }

    fn resolve(specs: &[&str], default: Option<&str>) -> Result<Option<RouteTable>, String> {
        let specs: Vec<String> = specs.iter().map(|s| s.to_string()).collect();
        let default = default.map(named);
        RouteTable::resolve(&specs, default.as_ref(), Duration::from_secs(1), lookup)
    }

    #[test]
    fn no_auto_model_is_no_table() {
        assert_eq!(resolve(&[], None), Ok(None));
    }

    #[test]
    fn what_auto_model_refuses() {
        let refused = |specs: &[&str], says: &str| {
            let error = resolve(specs, None).unwrap_err();
            assert!(error.contains(says), "{specs:?}: {error}");
        };
        refused(&["qwen3-4b"], "2 to 8 models");
        refused(&["custom"; 9], "2 to 8 models");
        refused(&["qwen3-4b", "AUTO"], "the router");
        refused(&["qwen3-4b", "=fast"], "a model name comes before");
        refused(&["qwen3-4b", "qwen3-80b"], "'qwen3-80b' is not a model");
        refused(&["qwen3-4b", "qwen3:4b=again"], "twice");
        let long = format!("qwen3-8b={}", "x".repeat(MAX_DESCRIPTION_CHARS + 1));
        refused(&["qwen3-4b", &long], "at most 400");
        refused(&["qwen3-4b", "qwen3-8b=a\0b"], "NUL");
    }

    /// The flag's text first, then the store's, then the catalog's with
    /// the size and domain, then the name; the last two warn.
    #[test]
    fn a_description_comes_from_the_flag_the_store_the_catalog_or_the_name() {
        let table = resolve(
            &[
                "qwen3-4b",
                "qwen3-8b",
                "custom",
                "vision=Images: photos, charts, scans = all of them",
            ],
            None,
        )
        .unwrap()
        .unwrap();
        let described: Vec<(&str, DescriptionSource)> = table
            .candidates
            .iter()
            .map(|c| (c.description.as_str(), c.source))
            .collect();
        assert_eq!(
            described,
            [
                ("Everyday chat.", DescriptionSource::Store),
                (
                    "Mid-size Qwen3. (about 8 billion parameters, domain general)",
                    DescriptionSource::Catalog
                ),
                ("custom", DescriptionSource::Name),
                (
                    "Images: photos, charts, scans = all of them",
                    DescriptionSource::Flag
                ),
            ]
        );
        let warned: Vec<bool> = described.iter().map(|(_, s)| s.warns()).collect();
        assert_eq!(warned, [false, true, true, false]);
        assert!(table.candidates[3].has_projector);
        // An empty description after `=` is none at all.
        let table = resolve(&["qwen3-4b=", "custom"], None).unwrap().unwrap();
        assert_eq!(table.candidates[0].source, DescriptionSource::Store);
    }

    /// The fallback is `--default-model` when it is a candidate, by any of
    /// its names, and the last candidate otherwise.
    #[test]
    fn the_fallback_is_the_default_model_or_the_last_candidate() {
        let specs = ["qwen3-4b", "qwen3-8b", "custom"];
        let fallback = |default| {
            resolve(&specs, default)
                .unwrap()
                .unwrap()
                .fallback()
                .name
                .clone()
        };
        assert_eq!(fallback(None), "custom");
        assert_eq!(fallback(Some("qwen3-4b")), "qwen3-4b");
        assert_eq!(fallback(Some("QWEN3-8B")), "qwen3-8b");
        assert_eq!(fallback(Some("qwen3-14b")), "custom", "not a candidate");
    }

    fn table() -> RouteTable {
        resolve(&["qwen3-4b=Small", "vision=Images", "qwen3-8b=Large"], None)
            .unwrap()
            .unwrap()
    }

    #[test]
    fn a_request_goes_only_to_models_that_can_read_it() {
        let table = table();
        let contexts = [4096, 4096, 32768];
        let text = RequestFacts {
            attachments: 0,
            chars: 350,
        };
        assert_eq!(eligible(&table, text, &contexts), (vec![0, 1, 2], vec![]));

        let image = RequestFacts {
            attachments: 1,
            chars: 350,
        };
        let (offered, excluded) = eligible(&table, image, &contexts);
        assert_eq!(offered, [1]);
        assert_eq!(
            excluded
                .iter()
                .map(|e| e.model.as_str())
                .collect::<Vec<_>>(),
            ["qwen3-4b", "qwen3-8b"]
        );
        assert!(
            excluded[0].why.contains("cannot read images"),
            "{excluded:?}"
        );

        // About 5,715 tokens: more than 4,096.
        let long = RequestFacts {
            attachments: 0,
            chars: 20_000,
        };
        let (offered, excluded) = eligible(&table, long, &contexts);
        assert_eq!(offered, [2]);
        assert!(
            excluded[0].why.contains("context of 4096 tokens"),
            "{excluded:?}"
        );
    }

    #[test]
    fn the_question_offers_the_candidates_in_order_with_their_descriptions() {
        let table = table();
        let Question::Choice {
            instructions,
            options,
        } = question(&table, &[0, 2])
        else {
            panic!("a choice");
        };
        assert_eq!(instructions, ROUTE_QUESTION);
        assert_eq!(
            options,
            [
                ("qwen3-4b".to_string(), "Small".to_string()),
                ("qwen3-8b".to_string(), "Large".to_string())
            ]
        );
        question(&table, &[0, 1, 2])
            .validate()
            .expect("a valid question");
    }

    /// A route the decision model decided between the first and the last
    /// candidate of `table()`.
    fn decided(table: &RouteTable) -> Route {
        Route {
            id: Uuid::nil(),
            model: "qwen3-4b".into(),
            reason: RouteReason::Decided,
            fallback: "qwen3-8b".into(),
            offered: vec![0, 2],
            resident: vec![true, false, true],
            probabilities: Some(vec![0.75, 0.25]),
            confidence: Some(0.5),
            excluded: vec![ExcludedCandidate {
                model: "vision".into(),
                why: "excluded for the test".into(),
            }],
            decision_model: Some("jev-style-0.8b".into()),
            decision_ms: 38.5,
            state: "Prompt:\nhi".into(),
            question: question(table, &[0, 2]),
            error: None,
        }
    }

    /// What a routed response carries under `eullm.route`: the model that
    /// answers and why, each offered candidate's probability by name in the
    /// order offered, and the route's id — the state and the question stay
    /// out of every response.
    #[test]
    fn a_routed_response_says_which_model_answers_and_why() {
        let table = table();
        let info = RouteInfo::new(&decided(&table), &table);
        assert_eq!(
            serde_json::to_string(&info).unwrap(),
            r#"{"requested":"auto","model":"qwen3-4b","reason":"decided","confidence":0.5,"probabilities":{"qwen3-4b":0.75,"qwen3-8b":0.25},"decision_model":"jev-style-0.8b","decision_ms":38.5,"id":"00000000-0000-0000-0000-000000000000"}"#
        );
        assert_eq!(
            info.reference(),
            RouteRef {
                id: Uuid::nil(),
                requested: "auto".into(),
                fallback: None,
            }
        );

        let mut undecided = decided(&table);
        undecided.reason = RouteReason::Timeout;
        undecided.probabilities = None;
        undecided.confidence = None;
        let json = serde_json::to_value(RouteInfo::new(&undecided, &table)).unwrap();
        assert_eq!(json["reason"], "timeout");
        assert!(json["probabilities"].is_null() && json["confidence"].is_null());
    }

    /// When the model chosen does not load, the fallback answers, and both
    /// the response and the generation's audit line say so.
    #[test]
    fn a_route_that_fell_back_says_why() {
        let table = table();
        let mut info = RouteInfo::new(&decided(&table), &table);
        info.fell_back("qwen3-8b", "out of memory".into());
        let json = serde_json::to_value(&info).unwrap();
        assert_eq!(json["model"], "qwen3-8b");
        assert_eq!(json["reason"], "load_failed");
        assert_eq!(
            info.reference().fallback.as_deref(),
            Some("load_failed: out of memory")
        );
    }

    /// The headers name the model, the reason and the route; a value no
    /// header can carry is left out, and the others still go.
    #[test]
    fn a_routed_response_has_its_headers() {
        let table = table();
        let info = RouteInfo::new(&decided(&table), &table);
        let headers: Vec<(String, String)> = info
            .headers()
            .into_iter()
            .map(|(name, value)| (name.to_string(), value.to_str().unwrap().to_string()))
            .collect();
        assert_eq!(
            headers,
            [
                ("x-eullm-model".to_string(), "qwen3-4b".to_string()),
                ("x-eullm-route".to_string(), "decided".to_string()),
                (
                    "x-eullm-route-id".to_string(),
                    "00000000-0000-0000-0000-000000000000".to_string()
                ),
            ]
        );

        let mut odd = decided(&table);
        odd.model = "line\nbreak".into();
        let names: Vec<HeaderName> = RouteInfo::new(&odd, &table)
            .headers()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(names, [ROUTE_HEADER, ROUTE_ID_HEADER]);
    }

    /// The digest of a conversation, exactly: what it is, the start of the
    /// system instructions, the two turns before the latest message — the
    /// user's start and the assistant's end — and the latest message whole.
    #[test]
    fn the_state_of_a_conversation() {
        let messages = vec![
            json!({ "role": "system", "content": "You are   a helpful\n assistant." }),
            json!({ "role": "user", "content": "Write a function that sorts a list." }),
            json!({ "role": "assistant", "content": "def sort(xs):\n    return sorted(xs)" }),
            json!({ "role": "user", "content": "and in Rust?" }),
        ];
        let tools = vec!["run_code".to_string(), "search".to_string()];
        let input = RouteInput::Chat {
            messages: &messages,
            tools,
        };
        assert_eq!(
            routing_state(&input),
            "Request to answer, with its context.\n\
             Conversation: 4 messages, 2 from the user. Attachments: none. Tools offered: \
             run_code, search.\n\
             System instructions (start): You are a helpful assistant.\n\
             Earlier turns (most recent last):\n\
             user (start): Write a function that sorts a list.\n\
             assistant (end): def sort(xs): return sorted(xs)\n\
             Latest message:\n\
             and in Rust?"
        );
    }

    /// OpenAI's array content: the text parts are read, an image counted.
    #[test]
    fn the_state_of_an_openai_request_with_an_image() {
        let messages = vec![json!({ "role": "user", "content": [
            { "type": "text", "text": "What is in" },
            { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
            { "type": "text", "text": "this picture?" },
        ] })];
        let input = RouteInput::Chat {
            messages: &messages,
            tools: Vec::new(),
        };
        assert_eq!(
            routing_state(&input),
            "Request to answer, with its context.\n\
             Conversation: 1 message, 1 from the user. Attachments: 1. Tools offered: none.\n\
             Latest message:\n\
             What is in\nthis picture?"
        );
        assert_eq!(
            RequestFacts::of(&input),
            RequestFacts {
                attachments: 1,
                chars: 24
            }
        );
    }

    #[test]
    fn the_state_of_a_prompt() {
        let body = json!({ "model": "auto", "prompt": "Why is the sky blue?\0" });
        let input = RouteInput::of(&body).expect("a prompt");
        assert_eq!(
            routing_state(&input),
            "Request to answer, with its context.\nPrompt:\nWhy is the sky blue?"
        );
    }

    /// A long latest message keeps its start and its end, and says how much
    /// was left out between them.
    #[test]
    fn a_long_message_keeps_its_head_and_its_tail() {
        let text = format!(
            "{}{}{}",
            "a".repeat(3000),
            "b".repeat(2000),
            "c".repeat(1000)
        );
        let state = routing_state(&RouteInput::Generate { prompt: &text });
        let expected = format!(
            "{}\n[… 2000 characters left out …]\n{}",
            "a".repeat(3000),
            "c".repeat(1000)
        );
        assert!(state.ends_with(&expected), "{}", &state[..200]);
        // Exactly the limit is whole.
        let whole = "é".repeat(LATEST_WHOLE_CHARS);
        let state = routing_state(&RouteInput::Generate { prompt: &whole });
        assert!(state.ends_with(&whole));
    }

    #[test]
    fn many_tools_are_listed_as_far_as_they_fit() {
        let tools: Vec<String> = (0..100).map(|n| format!("tool_number_{n}")).collect();
        let listed = tool_names(&tools);
        assert!(
            listed.starts_with("tool_number_0, tool_number_1"),
            "{listed}"
        );
        assert!(listed.ends_with(" more"), "{listed}");
        assert!(listed.chars().count() <= MAX_TOOL_NAMES_CHARS + 20);
    }

    use proptest::prelude::*;

    /// Any message a client can send: a role, and content as a string, an
    /// array of parts, or something else entirely.
    fn any_message() -> impl Strategy<Value = Value> {
        let content = prop_oneof![
            ".{0,9000}".prop_map(Value::String),
            proptest::collection::vec(".{0,3000}", 0..4).prop_map(|texts| {
                Value::Array(
                    texts
                        .into_iter()
                        .map(|t| json!({ "type": "text", "text": t }))
                        .chain(std::iter::once(json!({ "type": "image_url" })))
                        .collect(),
                )
            }),
            Just(Value::Null),
            any::<i64>().prop_map(|n| json!(n)),
        ];
        let role = prop_oneof![
            Just("user".to_string()),
            Just("assistant".to_string()),
            Just("system".to_string()),
            Just("tool".to_string()),
            ".{0,20}",
        ];
        (role, content).prop_map(|(role, content)| json!({ "role": role, "content": content }))
    }

    proptest! {
        // Requests of tens of thousands of characters each: fewer cases
        // than the default keep this test as quick as the others.
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Whatever a request holds, the digest is built — no panic on a
        /// character boundary, an empty conversation or a missing user —
        /// and stays within its bound.
        #[test]
        fn the_state_is_bounded_whatever_the_request(
            messages in proptest::collection::vec(any_message(), 0..8),
            tools in proptest::collection::vec(".{0,60}", 0..40),
            prompt in ".{0,9000}",
        ) {
            let chat = RouteInput::Chat { messages: &messages, tools };
            let state = routing_state(&chat);
            prop_assert!(state.chars().count() <= MAX_STATE_CHARS);
            prop_assert!(!state.contains('\0'));
            let generate = routing_state(&RouteInput::Generate { prompt: &prompt });
            prop_assert!(generate.chars().count() <= MAX_STATE_CHARS);
            prop_assert!(!generate.contains('\0'));
        }
    }
}
