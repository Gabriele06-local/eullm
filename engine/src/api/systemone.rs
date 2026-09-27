//! `POST /v1/systemone` — typed decisions about a state, in the request and
//! response shape of the System One API (TypeSafe's Jev), so a client
//! written for it can point its base URL at a local EuLLM: the same
//! principle as the Ollama and OpenAI endpoints.
//!
//! ```json
//! { "state": "Help! My payouts have been failing for 3 days.",
//!   "questions": {
//!     "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" },
//!     "team": { "type": "choice", "instructions": "Which team handles it?",
//!               "criteria": { "billing": "Payments, payouts", "tech": "Bugs" } },
//!     "severity": { "type": "score", "instructions": "How severe is it?",
//!                   "criteria": ["Cosmetic", "Degraded", "Blocking"] } } }
//! ```
//!
//! Every answer is read from the decision model's next-token distribution
//! (`inference::decision`), never generated. Next to the System One fields
//! each answer carries an `eullm` object with what the answer was derived
//! from — the full-vocabulary log-probability of every code, the
//! probabilities before calibration and the coverage — so a stored response
//! can be re-examined, or re-calibrated, later. `eullm` in the request picks
//! the calibration and evaluation mode; everything in it is optional.
//!
//! `model` may name any model the server can load. Absent, or a System One
//! model name (`jev-latest`, `jev-1.13.0`), it means the decision model this
//! server already has loaded — so a Jev client works without changes.

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{AppState, KeepAlive};
use crate::audit::{AuditEntry, AuditLogger, DecisionAnswerRecord, DecisionRecord};
use crate::inference::decision::{
    self, DecideOptions, Decision, DecisionError, EvalMode, EvalStats, Question,
};

type S = Arc<AppState>;

/// Highest `temperature` accepted. Temperature scaling fitted on real data
/// lands around 0.5–3; anything past this is a mistake, not a calibration.
const MAX_TEMPERATURE: f64 = 100.0;

/// A JSON object kept in its original key order. `serde_json`'s own map
/// sorts its keys — this crate does not enable `preserve_order`, which would
/// reorder every other response too — but here the order carries meaning:
/// options are lettered A, B, C in the order the client listed them, score
/// levels run from the first to the last, and answers come back in the
/// order the questions were asked.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OrderedMap<V>(pub(crate) Vec<(String, V)>);

impl<'de, V: Deserialize<'de>> Deserialize<'de> for OrderedMap<V> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MapVisitor<V>(PhantomData<V>);

        impl<'de, V: Deserialize<'de>> Visitor<'de> for MapVisitor<V> {
            type Value = OrderedMap<V>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry::<String, V>()? {
                    entries.push(entry);
                }
                Ok(OrderedMap(entries))
            }
        }

        deserializer.deserialize_map(MapVisitor(PhantomData))
    }
}

impl<V: Serialize> Serialize for OrderedMap<V> {
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl<V> OrderedMap<V> {
    fn zip<T: Into<V> + Copy>(labels: &[String], values: &[T]) -> Self {
        OrderedMap(
            labels
                .iter()
                .cloned()
                .zip(values.iter().map(|&v| v.into()))
                .collect(),
        )
    }
}

/// Any JSON value, with objects in their original key order: how a
/// structured `state`, or an option described by an object, is shown to
/// the model.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum OrderedJson {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<OrderedJson>),
    Object(OrderedMap<OrderedJson>),
}

impl<'de> Deserialize<'de> for OrderedJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JsonVisitor;

        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = OrderedJson;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_unit<E>(self) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Null)
            }
            fn visit_none<E>(self) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Null)
            }
            fn visit_bool<E>(self, v: bool) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Number(v.into()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Number(v.into()))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<OrderedJson, E> {
                serde_json::Number::from_f64(v)
                    .map(OrderedJson::Number)
                    .ok_or_else(|| E::custom("not a finite number"))
            }
            fn visit_str<E>(self, v: &str) -> Result<OrderedJson, E> {
                Ok(OrderedJson::String(v.to_string()))
            }
            fn visit_string<E>(self, v: String) -> Result<OrderedJson, E> {
                Ok(OrderedJson::String(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<OrderedJson, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(OrderedJson::Array(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<OrderedJson, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry::<String, OrderedJson>()? {
                    entries.push(entry);
                }
                Ok(OrderedJson::Object(OrderedMap(entries)))
            }
        }

        deserializer.deserialize_any(JsonVisitor)
    }
}

impl Serialize for OrderedJson {
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(b) => serializer.serialize_bool(*b),
            Self::Number(n) => n.serialize(serializer),
            Self::String(s) => serializer.serialize_str(s),
            Self::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Self::Object(map) => map.serialize(serializer),
        }
    }
}

impl OrderedJson {
    /// The text the model reads for a description: a string as it is,
    /// nothing for `null`, anything else as compact JSON in its original
    /// key order (a structured description such as
    /// `{"what": …, "not_for": …, "examples": […]}` reads best whole).
    fn as_description(&self) -> String {
        match self {
            Self::Null => String::new(),
            Self::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct SystemOneRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    state: Option<OrderedJson>,
    questions: OrderedMap<QuestionSpec>,
    #[serde(default)]
    keep_alive: Option<Value>,
    #[serde(default)]
    eullm: Option<RequestOptions>,
}

#[derive(Debug, Deserialize)]
struct QuestionSpec {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    criteria: Option<OrderedJson>,
}

/// The request's `eullm` object. Unknown keys are refused, so a typo in an
/// option name cannot silently leave a calibration off.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestOptions {
    /// `none` (default) or `content_free`.
    #[serde(default)]
    calibration: Option<String>,
    /// Temperature scaling applied after calibration; default 1.
    #[serde(default)]
    temperature: Option<f64>,
    /// `shared_prefix` (default), `batched` or `separate`.
    #[serde(default)]
    mode: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Calibration {
    None,
    ContentFree,
}

impl Calibration {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ContentFree => "content_free",
        }
    }
}

/// A request checked and turned into the engine's terms, before any model
/// is loaded — every client mistake is reported here.
#[derive(Debug)]
struct ParsedRequest {
    /// `None`: use the decision model already loaded.
    model: Option<String>,
    /// The state as the model reads it.
    state: String,
    ids: Vec<String>,
    questions: Vec<Question>,
    calibration: Calibration,
    temperature: f64,
    mode: EvalMode,
    keep_alive: KeepAlive,
}

fn parse_request(request: SystemOneRequest) -> Result<ParsedRequest, String> {
    let model = request
        .model
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty() && !m.to_ascii_lowercase().starts_with("jev"));

    let state = match request.state {
        None => return Err("\"state\" is required".to_string()),
        Some(OrderedJson::String(s)) => s,
        Some(other) => serde_json::to_string_pretty(&other).map_err(|e| e.to_string())?,
    };

    let specs = request.questions.0;
    if specs.is_empty() {
        return Err("\"questions\" must contain at least one question".to_string());
    }
    if specs.len() > decision::MAX_QUESTIONS {
        return Err(format!(
            "at most {} questions per request, got {}",
            decision::MAX_QUESTIONS,
            specs.len()
        ));
    }
    let mut ids = Vec::with_capacity(specs.len());
    let mut questions = Vec::with_capacity(specs.len());
    for (id, spec) in specs {
        if id.trim().is_empty() {
            return Err("question ids must not be empty".to_string());
        }
        if ids.contains(&id) {
            return Err(format!("duplicate question id \"{id}\""));
        }
        let question = parse_question(spec).map_err(|e| format!("question \"{id}\": {e}"))?;
        ids.push(id);
        questions.push(question);
    }

    let options = request.eullm.unwrap_or_default();
    let calibration = match options.calibration.as_deref() {
        None | Some("none") => Calibration::None,
        Some("content_free") => Calibration::ContentFree,
        Some(other) => {
            return Err(format!(
                "unknown calibration \"{other}\": expected \"none\" or \"content_free\""
            ));
        }
    };
    let temperature = options.temperature.unwrap_or(1.0);
    if !(temperature.is_finite() && temperature > 0.0 && temperature <= MAX_TEMPERATURE) {
        return Err(format!(
            "\"temperature\" must be greater than 0 and at most {MAX_TEMPERATURE}"
        ));
    }
    let mode = match options.mode.as_deref() {
        None | Some("shared_prefix") => EvalMode::SharedPrefix,
        Some("batched") => EvalMode::Batched,
        Some("separate") => EvalMode::Separate,
        Some(other) => {
            return Err(format!(
                "unknown mode \"{other}\": expected \"shared_prefix\", \"batched\" or \"separate\""
            ));
        }
    };

    Ok(ParsedRequest {
        model,
        state,
        ids,
        questions,
        calibration,
        temperature,
        mode,
        keep_alive: super::parse_keep_alive(request.keep_alive.as_ref()),
    })
}

fn parse_question(spec: QuestionSpec) -> Result<Question, String> {
    let instructions = spec
        .instructions
        .ok_or_else(|| "\"instructions\" is required".to_string())?;
    let question = match spec.kind.as_str() {
        "noul" => Question::Noul { instructions },
        "choice" => match spec.criteria {
            Some(OrderedJson::Object(options)) => Question::Choice {
                instructions,
                options: options
                    .0
                    .into_iter()
                    .map(|(name, description)| (name, description.as_description()))
                    .collect(),
            },
            _ => {
                return Err(
                    "a choice question needs \"criteria\": an object of option name → description"
                        .to_string(),
                );
            }
        },
        "score" => match spec.criteria {
            Some(OrderedJson::Array(levels)) => Question::Score {
                instructions,
                levels: levels.iter().map(OrderedJson::as_description).collect(),
            },
            _ => {
                return Err(
                    "a score question needs \"criteria\": an array of level descriptions, \
                     lowest first"
                        .to_string(),
                );
            }
        },
        other => {
            return Err(format!(
                "unknown question type \"{other}\": expected \"noul\", \"choice\" or \"score\""
            ));
        }
    };
    question.validate()?;
    Ok(question)
}

/// The answer labels of a question, in class order: how its probabilities
/// are keyed in the response.
fn labels(question: &Question) -> Vec<String> {
    match question {
        Question::Noul { .. } => vec!["yes".to_string(), "no".to_string()],
        Question::Choice { options, .. } => options.iter().map(|(name, _)| name.clone()).collect(),
        Question::Score { levels, .. } => (0..levels.len()).map(|i| i.to_string()).collect(),
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct SystemOneResponse {
    model: String,
    answers: OrderedMap<Answer>,
    usage: Usage,
    eullm: ResponseExtension,
}

/// One answer: the System One fields, then `eullm`.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Answer {
    Noul {
        /// P(yes).
        noul: f64,
        eullm: AnswerExtension,
    },
    Choice {
        choice: String,
        probabilities: OrderedMap<f64>,
        confidence: f64,
        eullm: AnswerExtension,
    },
    Score {
        /// Σ level × p(level).
        score: f64,
        legend: OrderedMap<String>,
        probabilities: OrderedMap<f64>,
        confidence: f64,
        eullm: AnswerExtension,
    },
}

/// What an answer was derived from.
#[derive(Debug, Serialize)]
struct AnswerExtension {
    /// Full-vocabulary log-probability of each answer's code.
    logprobs: OrderedMap<f64>,
    /// Renormalized over the answers, before any calibration.
    raw_probabilities: OrderedMap<f64>,
    /// Share of the model's probability on a valid answer code. Low means
    /// the model did not answer in the format asked for, and the
    /// probabilities describe a minority of what it would have said.
    coverage: f64,
    /// The content-free prior that was divided out (`content_free` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    prior_logprobs: Option<OrderedMap<f64>>,
}

#[derive(Debug, Serialize)]
struct Usage {
    /// Tokens actually decoded, content-free priors included.
    input_tokens: usize,
    /// Always 0: nothing is generated.
    output_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ResponseExtension {
    mode: &'static str,
    calibration: &'static str,
    temperature: f64,
    confidence_method: &'static str,
    /// `auto` or `off` (`--no-flash-attn`): which attention kernels the
    /// numbers below came from, since timings and the last digits of every
    /// probability depend on it.
    flash_attn: &'static str,
    /// All prompts' tokens together: what asking each question on its own
    /// decodes.
    prompt_tokens: usize,
    /// Tokens every prompt shares, decoded once.
    shared_prefix_tokens: usize,
    /// Those tokens were still in the decision model's context from the
    /// previous request, about the same state, and were not decoded again.
    prefix_reused: bool,
    /// Tokens decoded for the answers.
    evaluated_tokens: usize,
    timings_ms: Timings,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_free: Option<ContentFreeInfo>,
    /// Wall time of the whole request, model resolution included.
    request_ms: f64,
}

#[derive(Debug, Serialize)]
struct Timings {
    context: f64,
    prefix: f64,
    questions: f64,
    readout: f64,
}

impl From<&EvalStats> for Timings {
    fn from(s: &EvalStats) -> Self {
        Self {
            context: s.context_ms,
            prefix: s.prefix_ms,
            questions: s.questions_ms,
            readout: s.readout_ms,
        }
    }
}

#[derive(Debug, Serialize)]
struct ContentFreeInfo {
    /// Priors served from the cache.
    cached: usize,
    /// Tokens decoded for the priors that were not cached.
    evaluated_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    timings_ms: Option<Timings>,
}

/// Every answer, and its audit record, from the engine's log-probabilities.
fn build_answers(
    parsed: &ParsedRequest,
    decision: &Decision,
) -> (OrderedMap<Answer>, Vec<DecisionAnswerRecord>) {
    let mut answers = Vec::with_capacity(parsed.questions.len());
    let mut records = Vec::with_capacity(parsed.questions.len());
    for ((id, question), outcome) in parsed
        .ids
        .iter()
        .zip(&parsed.questions)
        .zip(&decision.outcomes)
    {
        let labels = labels(question);
        let raw = decision::calibrated_probabilities(&outcome.logprobs, None, 1.0);
        let prior = match parsed.calibration {
            Calibration::ContentFree => outcome.prior_logprobs.as_deref(),
            Calibration::None => None,
        };
        let probabilities =
            decision::calibrated_probabilities(&outcome.logprobs, prior, parsed.temperature);
        let coverage = decision::coverage(&outcome.logprobs);
        let extension = AnswerExtension {
            logprobs: OrderedMap::zip(&labels, &outcome.logprobs),
            raw_probabilities: OrderedMap::zip(&labels, &raw),
            coverage,
            prior_logprobs: prior.map(|p| OrderedMap::zip(&labels, p)),
        };

        let (answer, value, confidence) = match question {
            Question::Noul { .. } => {
                let noul = probabilities[0];
                (
                    Answer::Noul {
                        noul,
                        eullm: extension,
                    },
                    json!(noul),
                    None,
                )
            }
            Question::Choice { .. } => {
                let best = probabilities
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(i, _)| i);
                let confidence = decision::normalized_entropy_confidence(&probabilities);
                let choice = labels[best].clone();
                (
                    Answer::Choice {
                        choice: choice.clone(),
                        probabilities: OrderedMap::zip(&labels, &probabilities),
                        confidence,
                        eullm: extension,
                    },
                    json!(choice),
                    Some(confidence),
                )
            }
            Question::Score { levels, .. } => {
                let score = decision::expected_level(&probabilities);
                let confidence = decision::normalized_entropy_confidence(&probabilities);
                (
                    Answer::Score {
                        score,
                        legend: OrderedMap(
                            labels.iter().cloned().zip(levels.iter().cloned()).collect(),
                        ),
                        probabilities: OrderedMap::zip(&labels, &probabilities),
                        confidence,
                        eullm: extension,
                    },
                    json!(score),
                    Some(confidence),
                )
            }
        };
        records.push(DecisionAnswerRecord {
            id: id.clone(),
            kind: question.kind().as_str().to_string(),
            labels: labels.clone(),
            logprobs: outcome.logprobs.clone(),
            raw_probabilities: raw,
            probabilities,
            coverage,
            answer: value,
            confidence,
        });
        answers.push((id.clone(), answer));
    }
    (OrderedMap(answers), records)
}

fn error(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<Value>) {
    (status, Json(json!({ "error": message.into() })))
}

fn decision_error(e: DecisionError) -> (StatusCode, Json<Value>) {
    let status = match e {
        DecisionError::Invalid(_) | DecisionError::TooLong { .. } => StatusCode::BAD_REQUEST,
        DecisionError::Runtime(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    error(status, e.to_string())
}

fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `POST /v1/systemone`.
pub(super) async fn systemone(
    State(state): State<S>,
    // Present on every request: the auth middleware inserts an anonymous
    // identity when no keys are configured.
    axum::Extension(identity): axum::Extension<super::Identity>,
    body: Result<Json<SystemOneRequest>, JsonRejection>,
) -> Result<Json<SystemOneResponse>, (StatusCode, Json<Value>)> {
    let started = Instant::now();
    let Json(request) = body.map_err(|e| error(e.status(), e.body_text()))?;
    let parsed = parse_request(request).map_err(|e| error(StatusCode::BAD_REQUEST, e))?;

    let (model_name, model) = match parsed.model.as_deref() {
        Some(name) => {
            let model = state
                .ensure_decision_model(name)
                .await
                .map_err(|e| match e {
                    super::ModelError::NotFound(msg) => error(StatusCode::NOT_FOUND, msg),
                    super::ModelError::LoadFailed(msg) => {
                        error(StatusCode::INTERNAL_SERVER_ERROR, msg)
                    }
                })?;
            (name.to_string(), model)
        }
        None => {
            let slot = state.decision.read().await;
            let Some(slot) = slot.as_ref() else {
                return Err(error(
                    StatusCode::BAD_REQUEST,
                    "no decision model is loaded: start the server with --decision-model, \
                     or name a model in \"model\"",
                ));
            };
            (slot.model_name.clone(), slot.model.clone())
        }
    };
    state.touch_decision_slot(parsed.keep_alive).await;

    let options = DecideOptions {
        mode: parsed.mode,
        content_free: parsed.calibration == Calibration::ContentFree,
    };
    let flash_attn = if model.flash_attn() { "auto" } else { "off" };
    let (parsed, decision) = tokio::task::spawn_blocking(move || {
        let decision = model.decide(&parsed.state, &parsed.questions, options);
        (parsed, decision)
    })
    .await
    .map_err(|e| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Decision task failed: {e}"),
        )
    })?;
    let decision = decision.map_err(decision_error)?;

    let (answers, records) = build_answers(&parsed, &decision);
    let prior_tokens = decision
        .prior_stats
        .as_ref()
        .map_or(0, |s| s.evaluated_tokens);
    let input_tokens = decision.stats.evaluated_tokens + prior_tokens;
    let request_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut audit = AuditEntry::new(model_name.clone(), "systemone".to_string());
    audit.input_tokens = u32::try_from(input_tokens).unwrap_or(u32::MAX);
    audit.duration_ms = request_ms as u64;
    audit.user_id = identity.key_id().map(str::to_string);
    audit.decision = Some(DecisionRecord {
        state_sha256: sha256_hex(&parsed.state),
        mode: parsed.mode.as_str().to_string(),
        calibration: parsed.calibration.as_str().to_string(),
        temperature: parsed.temperature,
        answers: records,
    });
    AuditLogger::new().log(&audit);

    Ok(Json(SystemOneResponse {
        model: model_name,
        answers,
        usage: Usage {
            input_tokens,
            output_tokens: 0,
        },
        eullm: ResponseExtension {
            mode: decision.stats.mode.as_str(),
            calibration: parsed.calibration.as_str(),
            temperature: parsed.temperature,
            confidence_method: decision::CONFIDENCE_METHOD,
            flash_attn,
            prompt_tokens: decision.stats.prompt_tokens,
            shared_prefix_tokens: decision.stats.shared_prefix_tokens,
            prefix_reused: decision.stats.prefix_reused,
            evaluated_tokens: decision.stats.evaluated_tokens,
            timings_ms: Timings::from(&decision.stats),
            content_free: (parsed.calibration == Calibration::ContentFree).then(|| {
                ContentFreeInfo {
                    cached: decision.priors_cached,
                    evaluated_tokens: prior_tokens,
                    timings_ms: decision.prior_stats.as_ref().map(Timings::from),
                }
            }),
            request_ms,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::decision::QuestionOutcome;

    fn parse(body: Value) -> Result<ParsedRequest, String> {
        let request: SystemOneRequest =
            serde_json::from_value(body).map_err(|e| format!("deserialize: {e}"))?;
        parse_request(request)
    }

    /// A request as a client writes it. Parsed from text because
    /// `serde_json::json!` sorts object keys, and order is what several of
    /// these tests are about.
    const REQUEST: &str = r#"{
        "model": "jev-latest",
        "state": "Help! My payouts have been failing for 3 days.",
        "questions": {
            "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" },
            "team": { "type": "choice", "instructions": "Which team should handle it?",
                      "criteria": { "tech": "Bugs", "billing": "Payments and payouts", "other": null } },
            "severity": { "type": "score", "instructions": "How severe is it?",
                          "criteria": ["Cosmetic", "Degraded", "Blocking"] }
        },
        "eullm": EULLM
    }"#;

    fn parse_text(eullm: &str) -> ParsedRequest {
        let text = REQUEST.replace("EULLM", eullm);
        parse_request(serde_json::from_str(&text).expect("valid request")).expect("parses")
    }

    fn body() -> Value {
        json!({
            "model": "jev-latest",
            "state": "Help! My payouts have been failing for 3 days.",
            "questions": {
                "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" },
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle it?",
                    "criteria": { "tech": "Bugs", "billing": "Payments and payouts", "other": null }
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is it?",
                    "criteria": ["Cosmetic", "Degraded", "Blocking"]
                }
            }
        })
    }

    #[test]
    fn a_system_one_request_parses_in_the_order_it_was_written() {
        // `serde_json::json!` itself sorts object keys, so order is checked
        // on a request parsed from text, the way it arrives.
        let text = r#"{
            "state": "x",
            "questions": {
                "zeta": { "type": "noul", "instructions": "First?" },
                "alpha": { "type": "choice", "instructions": "Second?",
                           "criteria": { "tech": "Bugs", "billing": "Payments", "other": null } }
            }
        }"#;
        let request: SystemOneRequest = serde_json::from_str(text).unwrap();
        let parsed = parse_request(request).unwrap();
        assert_eq!(parsed.ids, ["zeta", "alpha"]);
        let Question::Choice { options, .. } = &parsed.questions[1] else {
            panic!("not a choice");
        };
        let names: Vec<&str> = options.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            ["tech", "billing", "other"],
            "A, B, C as the client listed them"
        );
        assert_eq!(options[2].1, "", "a null description is no description");
    }

    #[test]
    fn a_jev_model_name_means_the_loaded_decision_model() {
        assert_eq!(parse(body()).unwrap().model, None);
        let mut b = body();
        b["model"] = json!("JEV-1.13.0");
        assert_eq!(parse(b).unwrap().model, None);
        let mut b = body();
        b["model"] = json!("qwen3-4b");
        assert_eq!(parse(b).unwrap().model.as_deref(), Some("qwen3-4b"));
        let mut b = body();
        b.as_object_mut().unwrap().remove("model");
        assert_eq!(parse(b).unwrap().model, None);
    }

    #[test]
    fn defaults_are_no_calibration_shared_prefix_and_temperature_one() {
        let parsed = parse(body()).unwrap();
        assert_eq!(parsed.calibration, Calibration::None);
        assert_eq!(parsed.mode, EvalMode::SharedPrefix);
        assert_eq!(parsed.temperature, 1.0);
        assert_eq!(parsed.questions.len(), 3);
    }

    #[test]
    fn every_mode_is_asked_for_by_the_name_it_is_reported_under() {
        for mode in [
            EvalMode::SharedPrefix,
            EvalMode::Batched,
            EvalMode::Separate,
        ] {
            let mut b = body();
            b["eullm"] = json!({ "mode": mode.as_str() });
            assert_eq!(parse(b).unwrap().mode, mode);
        }
    }

    #[test]
    fn a_structured_state_is_shown_in_its_own_key_order() {
        let text = r#"{ "state": {"subject": "Refund", "body": "Charged twice", "amount": 12.5},
                        "questions": { "q": { "type": "noul", "instructions": "Refund?" } } }"#;
        let request: SystemOneRequest = serde_json::from_str(text).unwrap();
        let state = parse_request(request).unwrap().state;
        let subject = state.find("subject").unwrap();
        let body = state.find("body").unwrap();
        let amount = state.find("amount").unwrap();
        assert!(subject < body && body < amount, "{state}");
        assert!(state.contains("12.5"));
    }

    #[test]
    fn structured_option_descriptions_become_compact_json() {
        let text = r#"{ "state": "x", "questions": { "q": { "type": "choice", "instructions": "Which?",
            "criteria": { "billing": {"what": "payments", "not_for": "bugs"}, "tech": "bugs" } } } }"#;
        let request: SystemOneRequest = serde_json::from_str(text).unwrap();
        let parsed = parse_request(request).unwrap();
        let Question::Choice { options, .. } = &parsed.questions[0] else {
            panic!("not a choice");
        };
        assert_eq!(options[0].1, r#"{"what":"payments","not_for":"bugs"}"#);
    }

    #[test]
    fn client_mistakes_are_named() {
        let cases = [
            (
                json!({"questions": {"q": {"type": "noul", "instructions": "?"}}}),
                "\"state\" is required",
            ),
            (
                json!({"state": "x", "questions": {}}),
                "at least one question",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "maybe", "instructions": "?"}}}),
                "unknown question type",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "noul"}}}),
                "\"instructions\" is required",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}}}),
                "needs \"criteria\": an object",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "score", "instructions": "?", "criteria": {"a": "b"}}}}),
                "needs \"criteria\": an array",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["only one"]}}}),
                "2 to 10 levels",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}, "eullm": {"calibration": "platt"}}),
                "unknown calibration",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}, "eullm": {"mode": "fast"}}),
                "unknown mode",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}, "eullm": {"temperature": 0}}),
                "\"temperature\"",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}, "eullm": {"temprature": 2}}),
                "unknown field",
            ),
        ];
        for (body, expected) in cases {
            let err = parse(body.clone()).unwrap_err();
            assert!(err.contains(expected), "{body}: {err}");
        }
        // The question's id leads its own error, so a long request says
        // which question is wrong.
        let err =
            parse(json!({"state": "x", "questions": {"routing": {"type": "noul"}}})).unwrap_err();
        assert!(err.starts_with("question \"routing\""), "{err}");
    }

    #[test]
    fn too_many_questions_are_refused_before_any_model_is_loaded() {
        let questions: serde_json::Map<String, Value> = (0..=decision::MAX_QUESTIONS)
            .map(|i| {
                (
                    format!("q{i}"),
                    json!({"type": "noul", "instructions": "?"}),
                )
            })
            .collect();
        let err = parse(json!({"state": "x", "questions": questions})).unwrap_err();
        assert!(err.contains("at most 64"), "{err}");
    }

    #[test]
    fn duplicate_question_ids_are_refused() {
        let text = r#"{ "state": "x", "questions": {
            "q": { "type": "noul", "instructions": "One?" },
            "q": { "type": "noul", "instructions": "Two?" } } }"#;
        let request: SystemOneRequest = serde_json::from_str(text).unwrap();
        assert!(
            parse_request(request)
                .unwrap_err()
                .contains("duplicate question id")
        );
    }

    /// A decision whose log-probabilities put `p` (renormalized) on the
    /// classes, with 90% coverage.
    fn outcome(p: &[f64], prior: Option<&[f64]>) -> QuestionOutcome {
        QuestionOutcome {
            logprobs: p.iter().map(|v| (v * 0.9).ln()).collect(),
            prior_logprobs: prior.map(|q| q.iter().map(|v| v.ln()).collect()),
        }
    }

    fn fake_decision(outcomes: Vec<QuestionOutcome>) -> Decision {
        let stats = EvalStats {
            mode: EvalMode::SharedPrefix,
            prompts: outcomes.len(),
            shared_prefix_tokens: 100,
            evaluated_tokens: 160,
            prompt_tokens: 360,
            context_cells: 160,
            context_ms: 1.0,
            prefix_ms: 2.0,
            prefix_reused: false,
            questions_ms: 3.0,
            readout_ms: 0.5,
        };
        Decision {
            outcomes,
            stats,
            prior_stats: None,
            priors_cached: 0,
        }
    }

    #[test]
    fn answers_carry_the_system_one_fields_and_what_they_came_from() {
        let parsed = parse_text("{}");
        assert_eq!(parsed.ids, ["is_urgent", "team", "severity"]);
        let decision = fake_decision(vec![
            outcome(&[0.95, 0.05], None),
            outcome(&[0.1, 0.8, 0.1], None),
            outcome(&[0.0001, 0.57, 0.4299], None),
        ]);
        let (answers, records) = build_answers(&parsed, &decision);
        let json = serde_json::to_value(&answers).unwrap();

        let urgent = &json["is_urgent"];
        assert_eq!(urgent["type"], "noul");
        assert!((urgent["noul"].as_f64().unwrap() - 0.95).abs() < 1e-9);
        assert!((urgent["eullm"]["coverage"].as_f64().unwrap() - 0.9).abs() < 1e-9);
        assert!(urgent["eullm"]["logprobs"]["yes"].as_f64().unwrap() < 0.0);
        assert!(
            urgent.get("confidence").is_none(),
            "System One reports none for noul"
        );

        let team = &json["team"];
        assert_eq!(team["choice"], "billing");
        assert!((team["probabilities"]["billing"].as_f64().unwrap() - 0.8).abs() < 1e-9);
        assert!(team["confidence"].as_f64().unwrap() > 0.0);

        let severity = &json["severity"];
        assert!((severity["score"].as_f64().unwrap() - (0.57 + 2.0 * 0.4299)).abs() < 1e-6);
        assert_eq!(severity["legend"]["2"], "Blocking");
        assert!(severity["eullm"].get("prior_logprobs").is_none());

        assert_eq!(records.len(), 3);
        assert_eq!(records[1].answer, "billing");
        assert_eq!(records[1].labels, ["tech", "billing", "other"]);
        assert_eq!(records[0].confidence, None);
    }

    #[test]
    fn content_free_calibration_changes_the_answer_but_not_the_raw_record() {
        let parsed = parse_text(r#"{"calibration": "content_free"}"#);
        // The model says 0.75 yes about this state, but said 0.75 yes about
        // no state at all: calibrated, the state is no evidence either way.
        let decision = fake_decision(vec![
            outcome(&[0.75, 0.25], Some(&[0.75, 0.25])),
            outcome(&[0.1, 0.8, 0.1], Some(&[0.2, 0.6, 0.2])),
            outcome(&[0.2, 0.5, 0.3], Some(&[1.0 / 3.0; 3])),
        ]);
        let (answers, records) = build_answers(&parsed, &decision);
        let json = serde_json::to_value(&answers).unwrap();
        assert!((json["is_urgent"]["noul"].as_f64().unwrap() - 0.5).abs() < 1e-9);
        assert!(
            (json["is_urgent"]["eullm"]["raw_probabilities"]["yes"]
                .as_f64()
                .unwrap()
                - 0.75)
                .abs()
                < 1e-9
        );
        assert!(json["is_urgent"]["eullm"]["prior_logprobs"]["yes"].is_number());
        assert!((records[0].raw_probabilities[0] - 0.75).abs() < 1e-9);
        assert!((records[0].probabilities[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn the_response_serializes_answers_in_request_order() {
        let text = r#"{ "state": "x", "questions": {
            "zeta": { "type": "noul", "instructions": "?" },
            "alpha": { "type": "noul", "instructions": "?" } } }"#;
        let parsed = parse_request(serde_json::from_str(text).unwrap()).unwrap();
        let decision = fake_decision(vec![outcome(&[0.6, 0.4], None), outcome(&[0.3, 0.7], None)]);
        let (answers, _) = build_answers(&parsed, &decision);
        let text = serde_json::to_string(&answers).unwrap();
        assert!(
            text.find("zeta").unwrap() < text.find("alpha").unwrap(),
            "{text}"
        );
        assert!(
            text.starts_with(r#"{"zeta":{"type":"noul","noul":"#),
            "{text}"
        );
    }

    #[test]
    fn the_state_hash_is_sha256_hex() {
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
