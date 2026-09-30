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
use axum::response::{IntoResponse, Response};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{AppState, KeepAlive};
use crate::audit::{AuditEntry, AuditLogger, DecisionAnswerRecord, DecisionRecord};
use crate::inference::decision::{
    self, DecideOptions, Decision, DecisionError, EvalMode, EvalStats, Question, ReadoutKind,
};

type S = Arc<AppState>;

/// Where the endpoint is served: the middleware answers requests to it with
/// [`ApiError`]'s body.
pub(crate) const PATH: &str = "/v1/systemone";

/// Highest `temperature` accepted. Temperature scaling fitted on real data
/// lands around 0.5–3; anything past this is a mistake, not a calibration.
const MAX_TEMPERATURE: f64 = 100.0;

/// A `/v1/systemone` error as the System One API's clients and jev-style's
/// read it: an HTTP status, and the body
/// `{"error": {"code": …, "message": …, "question": …}}`, `question` naming
/// the question at fault when one is. The bare `{"error": "<message>"}` of
/// EuLLM's other endpoints made jev-style's MCP server and guard fail on
/// `err.get(…)`, so a client saw "Error executing tool" instead of what was
/// wrong.
///
/// The codes and their statuses are jev-style's where it has one:
/// `invalid_json`, `invalid_request`, `invalid_question` and
/// `input_budget_exceeded` with 422 — the status the System One API gives a
/// request that fails validation — `unauthorized` (401), `not_found` (404),
/// `method_not_allowed` (405) and `internal_error` (500). The rest are for
/// what jev-style's server never refuses: `model_not_loaded` (400),
/// `forbidden` (403), `payload_too_large` (413), `unsupported_media_type`
/// (415) and `too_many_requests` (429).
#[derive(Debug)]
pub(crate) struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    question: Option<String>,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            question: None,
        }
    }

    /// The request as a whole fails validation.
    fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", message)
    }

    /// Question `id` fails validation.
    fn invalid_question(id: &str, message: impl Into<String>) -> Self {
        Self::invalid_request(message).in_question(id)
    }

    /// This error as question `id`'s: named in `question`, and an invalid
    /// request narrowed to an invalid question.
    fn in_question(mut self, id: &str) -> Self {
        if self.code == "invalid_request" {
            self.code = "invalid_question";
        }
        self.question = Some(id.to_string());
        self
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
    }

    fn body(&self) -> Value {
        let mut error = json!({ "code": self.code, "message": self.message });
        if let Some(question) = &self.question {
            error["question"] = json!(question);
        }
        json!({ "error": error })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

/// A body axum could not read as a request: not JSON is `invalid_json`, the
/// wrong shape an `invalid_request`, both 422 as for any other validation
/// failure.
fn rejection(e: JsonRejection) -> ApiError {
    let message = e.body_text();
    match e {
        JsonRejection::JsonSyntaxError(_) => {
            ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_json", message)
        }
        JsonRejection::JsonDataError(_) => ApiError::invalid_request(message),
        JsonRejection::MissingJsonContentType(_) => ApiError::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "unsupported_media_type",
            message,
        ),
        other if other.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large", message)
        }
        other => ApiError::new(other.status(), "invalid_request", message),
    }
}

/// Any method but `POST` on the endpoint: the 405 axum answers, with the
/// body the endpoint's clients read.
pub(super) async fn method_not_allowed() -> ApiError {
    ApiError::new(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        format!("{PATH} accepts POST only"),
    )
}

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
    /// The value of `key`: the last one given, as a JSON parser that keeps
    /// one value per key reads it.
    fn get(&self, key: &str) -> Option<&V> {
        self.0.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v)
    }

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
    /// nothing for `null`, anything else as one line of JSON in its original
    /// key order (a structured description such as
    /// `{"what": …, "not_for": …, "examples": […]}` reads best whole).
    fn as_description(&self) -> String {
        match self {
            Self::Null => String::new(),
            Self::String(s) => s.clone(),
            other => other.python_json(),
        }
    }

    /// One line of JSON as Python's `json.dumps(value, ensure_ascii=False)`
    /// writes it — `", "` and `": "` between items, non-ASCII as it is,
    /// floats as `repr` spells them — which is how Jev-Style models saw
    /// structured states and descriptions in training.
    fn python_json(&self) -> String {
        self.json_line(", ", ": ")
    }

    /// The same with `separators=(",", ":")`: jev-style's `compact_json`,
    /// how its server writes structured instructions for the model and
    /// names a structured score level in the legend.
    fn compact_json(&self) -> String {
        self.json_line(",", ":")
    }

    /// One line of JSON with `item` between items and `key` after a key.
    fn json_line(&self, item: &str, key: &str) -> String {
        let mut out = String::new();
        self.write_json(&mut out, item, key);
        out
    }

    fn write_json(&self, out: &mut String, item: &str, key: &str) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Number(n) => out.push_str(&python_number(n)),
            // Escaped as Python escapes with `ensure_ascii=False`: quotes,
            // backslashes and control characters only.
            Self::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
            Self::Array(items) => {
                out.push('[');
                for (i, value) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    value.write_json(out, item, key);
                }
                out.push(']');
            }
            Self::Object(map) => {
                out.push('{');
                for (i, (name, value)) in map.0.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    out.push_str(&serde_json::to_string(name).unwrap_or_default());
                    out.push_str(key);
                    value.write_json(out, item, key);
                }
                out.push('}');
            }
        }
    }
}

/// A score level as the model reads it, and its name in the legend.
///
/// A string is both. A `{"label": …, "description": …}` object — how
/// jev-style's guard writes its risk scale — reads `"label: description"`,
/// or the label alone without a description, and the legend names it by its
/// label: what jev-style's own server does (`schema._level_text` and
/// `_level_label`), and how its training data writes levels. Such an object
/// with keys of its own besides, or any other object or array, is shown to
/// the model as JSON, like a structured option; the legend then names it by
/// its label if it has one, by its compact JSON otherwise. It used to be
/// JSON everywhere: the model read `{"label": "calm", "description": …}`
/// and the legend repeated it, where a client looks for `calm`.
fn score_level(i: usize, level: &OrderedJson) -> Result<(String, String), String> {
    match level {
        OrderedJson::String(text) => Ok((text.clone(), text.clone())),
        OrderedJson::Object(fields) if fields.get("label").is_some() => {
            let label = match fields.get("label") {
                Some(OrderedJson::String(label)) if !label.trim().is_empty() => label.clone(),
                _ => {
                    return Err(format!(
                        "score level {i}: \"label\" must be a non-empty string"
                    ));
                }
            };
            let description = match fields.get("description") {
                None | Some(OrderedJson::Null) => "",
                Some(OrderedJson::String(description)) => description.as_str(),
                Some(_) => {
                    return Err(format!("score level {i}: \"description\" must be a string"));
                }
            };
            let more = fields
                .0
                .iter()
                .any(|(key, _)| key != "label" && key != "description");
            let text = if more {
                level.python_json()
            } else if description.is_empty() {
                label.clone()
            } else {
                format!("{label}: {description}")
            };
            Ok((text, label))
        }
        OrderedJson::Object(OrderedMap(fields)) if !fields.is_empty() => {
            Ok((level.python_json(), level.compact_json()))
        }
        OrderedJson::Array(items) if !items.is_empty() => {
            Ok((level.python_json(), level.compact_json()))
        }
        _ => Err(format!(
            "score level {i} must be a non-empty string, a {{\"label\", \"description\"}} \
             object, or another non-empty object or array"
        )),
    }
}

/// A JSON number as Python writes it: integers as they are, floats as
/// `repr` — positional from 1e-4 to 1e16 with at least one decimal, else
/// `d.ddde±XX`, always the shortest digits that read back exactly.
fn python_number(n: &serde_json::Number) -> String {
    if !n.is_f64() {
        return n.to_string();
    }
    let x = n.as_f64().unwrap_or_default();
    // `{:e}` gives the shortest round-trip digits: "-1.25e1", "1e16".
    let scientific = format!("{x:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits = mantissa.replace('.', "");
    let body = if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        }
    } else {
        let mantissa = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{exp_sign}{:02}", exponent.unsigned_abs())
    };
    format!("{sign}{body}")
}

#[derive(Debug, Deserialize)]
pub(crate) struct SystemOneRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    state: Option<OrderedJson>,
    /// Each question as it was written, read by [`QuestionSpec::read`]:
    /// whatever is wrong inside one is then that question's error, named
    /// in the response's `question`.
    questions: OrderedMap<OrderedJson>,
    #[serde(default)]
    keep_alive: Option<Value>,
    #[serde(default)]
    eullm: Option<RequestOptions>,
}

/// One question as the request wrote it.
#[derive(Debug)]
struct QuestionSpec {
    kind: String,
    instructions: Option<OrderedJson>,
    criteria: Option<OrderedJson>,
}

impl QuestionSpec {
    /// The fields of a question object. Any other key is ignored, as the
    /// System One API and jev-style ignore it.
    fn read(question: OrderedJson) -> Result<Self, String> {
        let OrderedJson::Object(fields) = question else {
            return Err("a question must be an object with \"type\" and \"instructions\"".into());
        };
        let (mut kind, mut instructions, mut criteria) = (None, None, None);
        for (key, value) in fields.0 {
            let field = match key.as_str() {
                "type" => &mut kind,
                "instructions" => &mut instructions,
                "criteria" => &mut criteria,
                _ => continue,
            };
            if field.replace(value).is_some() {
                return Err(format!("duplicate field \"{key}\""));
            }
        }
        let kind = match kind {
            Some(OrderedJson::String(kind)) => kind,
            None | Some(OrderedJson::Null) => {
                return Err("\"type\" is required: \"noul\", \"choice\" or \"score\"".into());
            }
            Some(_) => return Err("\"type\" must be \"noul\", \"choice\" or \"score\"".into()),
        };
        Ok(Self {
            kind,
            instructions,
            criteria,
        })
    }
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
    /// The state as a code-readout model reads it: a string as it is,
    /// structured state as indented JSON.
    state: String,
    /// The state as a verdict model reads it: structured state as the one
    /// line of JSON it was trained on.
    state_line: String,
    ids: Vec<String>,
    questions: Vec<Question>,
    /// Per question, what each score level is called in the response's
    /// `legend`; empty for the other types.
    legends: Vec<Vec<String>>,
    calibration: Calibration,
    /// `None`: the model's own default (`DecisionModel::default_temperature`).
    temperature: Option<f64>,
    mode: EvalMode,
    keep_alive: KeepAlive,
}

fn parse_request(request: SystemOneRequest) -> Result<ParsedRequest, ApiError> {
    let model = request
        .model
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty() && !m.to_ascii_lowercase().starts_with("jev"));

    let (state, state_line) = match request.state {
        None => return Err(ApiError::invalid_request("\"state\" is required")),
        Some(OrderedJson::String(s)) => (s.clone(), s),
        Some(other) => (
            serde_json::to_string_pretty(&other).map_err(|e| ApiError::internal(e.to_string()))?,
            other.python_json(),
        ),
    };

    let specs = request.questions.0;
    if specs.is_empty() {
        return Err(ApiError::invalid_request(
            "\"questions\" must contain at least one question",
        ));
    }
    if specs.len() > decision::MAX_QUESTIONS {
        return Err(ApiError::invalid_request(format!(
            "at most {} questions per request, got {}",
            decision::MAX_QUESTIONS,
            specs.len()
        )));
    }
    let mut ids = Vec::with_capacity(specs.len());
    let mut questions = Vec::with_capacity(specs.len());
    let mut legends = Vec::with_capacity(specs.len());
    for (id, spec) in specs {
        if id.trim().is_empty() {
            return Err(ApiError::invalid_request("question ids must not be empty"));
        }
        if ids.contains(&id) {
            return Err(ApiError::invalid_question(&id, "duplicate question id"));
        }
        let (question, legend) = QuestionSpec::read(spec)
            .and_then(parse_question)
            .map_err(|e| ApiError::invalid_question(&id, e))?;
        ids.push(id);
        questions.push(question);
        legends.push(legend);
    }

    let options = request.eullm.unwrap_or_default();
    let calibration = match options.calibration.as_deref() {
        None | Some("none") => Calibration::None,
        Some("content_free") => Calibration::ContentFree,
        Some(other) => {
            return Err(ApiError::invalid_request(format!(
                "unknown calibration \"{other}\": expected \"none\" or \"content_free\""
            )));
        }
    };
    let temperature = options.temperature;
    if temperature.is_some_and(|t| !(t.is_finite() && t > 0.0 && t <= MAX_TEMPERATURE)) {
        return Err(ApiError::invalid_request(format!(
            "\"temperature\" must be greater than 0 and at most {MAX_TEMPERATURE}"
        )));
    }
    let mode = match options.mode.as_deref() {
        None | Some("shared_prefix") => EvalMode::SharedPrefix,
        Some("batched") => EvalMode::Batched,
        Some("separate") => EvalMode::Separate,
        Some(other) => {
            return Err(ApiError::invalid_request(format!(
                "unknown mode \"{other}\": expected \"shared_prefix\", \"batched\" or \"separate\""
            )));
        }
    };

    Ok(ParsedRequest {
        model,
        state,
        state_line,
        ids,
        questions,
        legends,
        calibration,
        temperature,
        mode,
        keep_alive: super::parse_keep_alive(request.keep_alive.as_ref()),
    })
}

/// Instructions given as an object or an array, as the System One API
/// allows — the question in one field, the data it refers to in others —
/// in the text the model reads: compact JSON in the request's key order,
/// what jev-style's server hands its models (`schema.compact_json`,
/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"))`).
fn structured_instructions(value: &OrderedJson) -> Result<String, String> {
    match value {
        OrderedJson::Object(OrderedMap(fields)) if fields.is_empty() => {
            Err("\"instructions\" must not be an empty object".to_string())
        }
        OrderedJson::Array(items) if items.is_empty() => {
            Err("\"instructions\" must not be an empty array".to_string())
        }
        OrderedJson::Object(_) | OrderedJson::Array(_) => Ok(value.compact_json()),
        _ => Err("\"instructions\" must be a string, an object or an array".to_string()),
    }
}

/// A question in the engine's terms, and its score levels' names in the
/// legend (none for the other types).
fn parse_question(spec: QuestionSpec) -> Result<(Question, Vec<String>), String> {
    let instructions = match spec.instructions {
        None | Some(OrderedJson::Null) => return Err("\"instructions\" is required".to_string()),
        Some(OrderedJson::String(instructions)) => instructions,
        Some(structured) => structured_instructions(&structured)?,
    };
    let mut legend = Vec::new();
    let question = match spec.kind.as_str() {
        "noul" => {
            let (mut true_means, mut false_means) = (String::new(), String::new());
            match spec.criteria {
                None | Some(OrderedJson::Null) => {}
                Some(OrderedJson::Object(outcomes)) => {
                    for (outcome, description) in outcomes.0 {
                        match outcome.as_str() {
                            "true" => true_means = description.as_description(),
                            "false" => false_means = description.as_description(),
                            other => {
                                return Err(format!(
                                    "a noul question's \"criteria\" says what \"true\" and \
                                     \"false\" mean, not \"{other}\""
                                ));
                            }
                        }
                    }
                }
                Some(_) => {
                    return Err("a noul question's \"criteria\", when given, is an object \
                         {\"true\": …, \"false\": …}"
                        .to_string());
                }
            }
            Question::Noul {
                instructions,
                true_means,
                false_means,
            }
        }
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
            Some(OrderedJson::Array(levels)) => {
                let (levels, names) = levels
                    .iter()
                    .enumerate()
                    .map(|(i, level)| score_level(i, level))
                    .collect::<Result<(Vec<_>, Vec<_>), _>>()?;
                legend = names;
                Question::Score {
                    instructions,
                    levels,
                }
            }
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
    Ok((question, legend))
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
    timing: ResponseTiming,
    eullm: ResponseExtension,
}

/// The response's `timing`, as jev-style's server reports it and its MCP
/// tools and guard read it (`total_ms`, shown next to every answer).
#[derive(Debug, Serialize)]
struct ResponseTiming {
    /// The request's wall time: `eullm.request_ms`, to the 0.1 ms
    /// jev-style rounds it to.
    total_ms: f64,
}

impl ResponseTiming {
    fn new(request_ms: f64) -> Self {
        Self {
            total_ms: (request_ms * 10.0).round() / 10.0,
        }
    }
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
    /// Code readout: full-vocabulary log-probability of each answer's code.
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<OrderedMap<f64>>,
    /// Verdict readout: `logit(" yes") - logit(" no")` at each answer's
    /// slot.
    #[serde(skip_serializing_if = "Option::is_none")]
    scores: Option<OrderedMap<f64>>,
    /// Renormalized over the answers (a softmax of the scores), before any
    /// calibration or temperature.
    raw_probabilities: OrderedMap<f64>,
    /// Code readout: share of the model's probability on a valid answer
    /// code. Low means the model did not answer in the format asked for,
    /// and the probabilities describe a minority of what it would have
    /// said. A verdict model answers every option by construction.
    #[serde(skip_serializing_if = "Option::is_none")]
    coverage: Option<f64>,
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
    /// `codes` or `verdict`: how the model's answers were read.
    readout: &'static str,
    mode: &'static str,
    calibration: &'static str,
    /// Applied to every answer: the request's, or the model's default —
    /// a verdict model's own calibration, 1 otherwise.
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
    readout: ReadoutKind,
    temperature: f64,
) -> (OrderedMap<Answer>, Vec<DecisionAnswerRecord>) {
    let mut answers = Vec::with_capacity(parsed.questions.len());
    let mut records = Vec::with_capacity(parsed.questions.len());
    for (((id, question), legend), outcome) in parsed
        .ids
        .iter()
        .zip(&parsed.questions)
        .zip(&parsed.legends)
        .zip(&decision.outcomes)
    {
        let labels = labels(question);
        let raw = decision::calibrated_probabilities(&outcome.logprobs, None, 1.0);
        let prior = match parsed.calibration {
            Calibration::ContentFree => outcome.prior_logprobs.as_deref(),
            Calibration::None => None,
        };
        let probabilities =
            decision::calibrated_probabilities(&outcome.logprobs, prior, temperature);
        let codes = readout == ReadoutKind::Codes;
        let coverage = codes.then(|| decision::coverage(&outcome.logprobs));
        let values = OrderedMap::zip(&labels, &outcome.logprobs);
        let (logprobs, scores) = if codes {
            (Some(values), None)
        } else {
            (None, Some(values))
        };
        let extension = AnswerExtension {
            logprobs,
            scores,
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
            Question::Score { .. } => {
                let score = decision::expected_level(&probabilities);
                let confidence = decision::normalized_entropy_confidence(&probabilities);
                (
                    Answer::Score {
                        score,
                        legend: OrderedMap(
                            labels.iter().cloned().zip(legend.iter().cloned()).collect(),
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
        let (logprobs, scores) = if codes {
            (Some(outcome.logprobs.clone()), None)
        } else {
            (None, Some(outcome.logprobs.clone()))
        };
        records.push(DecisionAnswerRecord {
            id: id.clone(),
            kind: question.kind().as_str().to_string(),
            labels: labels.clone(),
            logprobs,
            scores,
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

/// Why the engine could not decide, as the API reports it; `ids` names the
/// question when the error was one question's. Input over a budget is
/// `input_budget_exceeded`, as jev-style reports it: the request was not
/// cut to fit, and says what to shorten.
fn decision_error(e: DecisionError, ids: &[String]) -> ApiError {
    match e {
        DecisionError::Question(i, e) => {
            let error = decision_error(*e, ids);
            match ids.get(i) {
                Some(id) => error.in_question(id),
                None => error,
            }
        }
        DecisionError::Invalid(message) => ApiError::invalid_request(message),
        e @ (DecisionError::TooLong { .. } | DecisionError::OverBudget(_)) => ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "input_budget_exceeded",
            e.to_string(),
        ),
        DecisionError::Runtime(message) => ApiError::internal(message),
    }
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
) -> Result<Json<SystemOneResponse>, ApiError> {
    let started = Instant::now();
    let Json(request) = body.map_err(rejection)?;
    let parsed = parse_request(request)?;

    let (model_name, model) = match parsed.model.as_deref() {
        Some(name) => {
            let model = state
                .ensure_decision_model(name)
                .await
                .map_err(|e| match e {
                    super::ModelError::NotFound(msg) => {
                        ApiError::new(StatusCode::NOT_FOUND, "not_found", msg)
                    }
                    super::ModelError::LoadFailed(msg) => ApiError::internal(msg),
                })?;
            (name.to_string(), model)
        }
        None => {
            let slot = state.decision.read().await;
            let Some(slot) = slot.as_ref() else {
                // Not a validation failure: the request is well formed and
                // names, or implies, the decision model the server was
                // meant to have — 400, which a System One SDK does not retry.
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "model_not_loaded",
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
    let readout = model.readout();
    let temperature = parsed
        .temperature
        .unwrap_or_else(|| model.default_temperature());
    let (parsed, decision) = tokio::task::spawn_blocking(move || {
        let state = match readout {
            ReadoutKind::Codes => &parsed.state,
            ReadoutKind::Verdict => &parsed.state_line,
        };
        let decision = model.decide(state, &parsed.questions, options);
        (parsed, decision)
    })
    .await
    .map_err(|e| ApiError::internal(format!("Decision task failed: {e}")))?;
    let decision = decision.map_err(|e| decision_error(e, &parsed.ids))?;

    let (answers, records) = build_answers(&parsed, &decision, readout, temperature);
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
        state_sha256: sha256_hex(match readout {
            ReadoutKind::Codes => &parsed.state,
            ReadoutKind::Verdict => &parsed.state_line,
        }),
        readout: readout.as_str().to_string(),
        mode: decision.stats.mode.as_str().to_string(),
        calibration: parsed.calibration.as_str().to_string(),
        temperature,
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
        timing: ResponseTiming::new(request_ms),
        eullm: ResponseExtension {
            readout: readout.as_str(),
            mode: decision.stats.mode.as_str(),
            calibration: parsed.calibration.as_str(),
            temperature,
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

    /// A request as the handler reads it: a body that does not deserialize
    /// is the `invalid_request` axum's rejection becomes.
    fn parse(body: Value) -> Result<ParsedRequest, ApiError> {
        let request: SystemOneRequest = serde_json::from_value(body)
            .map_err(|e| ApiError::invalid_request(format!("deserialize: {e}")))?;
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
        assert_eq!(parsed.temperature, None);
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
    fn structured_option_descriptions_become_one_line_of_json() {
        let text = r#"{ "state": "x", "questions": { "q": { "type": "choice", "instructions": "Which?",
            "criteria": { "billing": {"what": "payments", "not_for": "bugs"}, "tech": "bugs" } } } }"#;
        let request: SystemOneRequest = serde_json::from_str(text).unwrap();
        let parsed = parse_request(request).unwrap();
        let Question::Choice { options, .. } = &parsed.questions[0] else {
            panic!("not a choice");
        };
        assert_eq!(options[0].1, r#"{"what": "payments", "not_for": "bugs"}"#);
    }

    /// The exact text Python's `json.dumps(value, ensure_ascii=False)`
    /// gives, which is what a Jev-Style model read in training.
    #[test]
    fn structured_values_are_written_as_python_writes_them() {
        let value: OrderedJson = serde_json::from_str(
            r#"{"b": [1, 2.5, 12.0, true, null], "a": {"é": "x\ny\"z", "t": "\u001b"},
                "big": 1e16, "small": 0.00001, "neg": -0.0, "int": -7, "mid": 1e15, "tiny": 0.0001}"#,
        )
        .unwrap();
        assert_eq!(
            value.python_json(),
            r#"{"b": [1, 2.5, 12.0, true, null], "a": {"é": "x\ny\"z", "t": "\u001b"}, "big": 1e+16, "small": 1e-05, "neg": -0.0, "int": -7, "mid": 1000000000000000.0, "tiny": 0.0001}"#
        );
    }

    #[test]
    fn a_structured_state_has_a_form_for_each_readout() {
        let text = r#"{ "state": {"subject": "Refund", "amount": 12.5},
                        "questions": { "q": { "type": "noul", "instructions": "Refund?" } } }"#;
        let parsed = parse_request(serde_json::from_str(text).unwrap()).unwrap();
        assert!(parsed.state.contains("\n  \"subject\""), "{}", parsed.state);
        assert_eq!(
            parsed.state_line,
            r#"{"subject": "Refund", "amount": 12.5}"#
        );
        let plain = parse_text("{}");
        assert_eq!(plain.state, plain.state_line);
    }

    /// Structured instructions reach the model as jev-style's server
    /// writes them: `json.dumps(v, ensure_ascii=False, separators=(",",
    /// ":"))`, keys in the order given.
    #[test]
    fn instructions_may_be_an_object_or_an_array() {
        let text = r#"{ "state": "x", "questions": {
            "dup": { "type": "noul", "instructions": {
                "potential_duplicate": {"name": "John Smith", "city": "Oakland", "age": 41.0},
                "question": "Is the resume for the same person as `potential_duplicate`?",
                "note": "line\nbreak \"quoted\" é" } },
            "list": { "type": "choice", "instructions": ["Which team?", {"hint": "billing first"}],
                      "criteria": {"billing": null, "tech": null} } } }"#;
        let parsed = parse_request(serde_json::from_str(text).unwrap()).unwrap();
        assert_eq!(
            parsed.questions[0].instructions(),
            r#"{"potential_duplicate":{"name":"John Smith","city":"Oakland","age":41.0},"question":"Is the resume for the same person as `potential_duplicate`?","note":"line\nbreak \"quoted\" é"}"#
        );
        assert_eq!(
            parsed.questions[1].instructions(),
            r#"["Which team?",{"hint":"billing first"}]"#
        );

        for (instructions, expected) in [
            ("{}", "must not be an empty object"),
            ("[]", "must not be an empty array"),
            ("5", "must be a string, an object or an array"),
            ("true", "must be a string, an object or an array"),
            (r#""  ""#, "must not be empty"),
            ("null", "is required"),
        ] {
            let text = format!(
                r#"{{ "state": "x", "questions": {{ "q": {{ "type": "noul",
                    "instructions": {instructions} }} }} }}"#
            );
            let err = parse_request(serde_json::from_str(&text).unwrap()).unwrap_err();
            assert_eq!(err.code, "invalid_question", "{instructions}");
            assert!(err.message.contains(expected), "{instructions}: {err:?}");
        }
    }

    /// Score levels written as jev-style's guard writes its risk scale:
    /// the model reads what jev-style's server gives its renderer, and the
    /// legend names each level as that server's legend does.
    #[test]
    fn score_levels_may_be_labels_with_descriptions() {
        let text = r#"{ "state": "rm -rf build", "questions": { "risk": { "type": "score",
            "instructions": "How risky is it?",
            "criteria": [
                {"label": "none", "description": "read-only"},
                {"label": "low"},
                {"label": "moderate", "description": ""},
                {"label": "high", "description": null},
                "severe",
                {"label": "tagged", "description": "d", "weight": 3},
                {"what": "odd", "n": 1.5},
                ["a", 2]
            ] } } }"#;
        let parsed = parse_request(serde_json::from_str(text).unwrap()).unwrap();
        let Question::Score { levels, .. } = &parsed.questions[0] else {
            panic!("not a score");
        };
        assert_eq!(
            levels,
            &[
                "none: read-only",
                "low",
                "moderate",
                "high",
                "severe",
                r#"{"label": "tagged", "description": "d", "weight": 3}"#,
                r#"{"what": "odd", "n": 1.5}"#,
                r#"["a", 2]"#,
            ]
        );
        assert_eq!(
            parsed.legends[0],
            [
                "none",
                "low",
                "moderate",
                "high",
                "severe",
                "tagged",
                r#"{"what":"odd","n":1.5}"#,
                r#"["a",2]"#,
            ]
        );

        let decision = fake_decision(vec![outcome(&[0.125; 8], None)]);
        let (answers, _) = build_answers(&parsed, &decision, ReadoutKind::Verdict, 1.0);
        let legend = &serde_json::to_value(&answers).unwrap()["risk"]["legend"];
        assert_eq!(legend["0"], "none");
        assert_eq!(legend["5"], "tagged");
    }

    #[test]
    fn a_score_level_that_names_nothing_is_refused() {
        for (level, expected) in [
            (r#"{"label": ""}"#, "\"label\" must be a non-empty string"),
            (r#"{"label": 3}"#, "\"label\" must be a non-empty string"),
            (
                r#"{"label": "a", "description": 3}"#,
                "\"description\" must be a string",
            ),
            ("{}", "must be a non-empty string"),
            ("[]", "must be a non-empty string"),
            ("3", "must be a non-empty string"),
            ("null", "must be a non-empty string"),
            (r#""  ""#, "must not be empty"),
        ] {
            let text = format!(
                r#"{{ "state": "x", "questions": {{ "q": {{ "type": "score", "instructions": "?",
                    "criteria": ["fine", {level}] }} }} }}"#
            );
            let err = parse_request(serde_json::from_str(&text).unwrap()).unwrap_err();
            assert_eq!(err.code, "invalid_question", "{level}");
            assert!(err.message.contains(expected), "{level}: {err:?}");
        }
    }

    #[test]
    fn a_noul_question_may_say_what_true_and_false_mean() {
        let text = r#"{ "state": "x", "questions": { "q": { "type": "noul", "instructions": "Happy?",
            "criteria": { "true": "says so", "false": {"else": "anything"} } } } }"#;
        let parsed = parse_request(serde_json::from_str(text).unwrap()).unwrap();
        let Question::Noul {
            true_means,
            false_means,
            ..
        } = &parsed.questions[0]
        else {
            panic!("not a noul");
        };
        assert_eq!(true_means, "says so");
        assert_eq!(false_means, r#"{"else": "anything"}"#);
        for bad in [r#"{"maybe": "x"}"#, r#"["x"]"#] {
            let text = format!(
                r#"{{ "state": "x", "questions": {{ "q": {{ "type": "noul", "instructions": "?", "criteria": {bad} }} }} }}"#
            );
            let err = parse_request(serde_json::from_str(&text).unwrap()).unwrap_err();
            assert!(err.message.contains("criteria"), "{err:?}");
        }
    }

    #[test]
    fn client_mistakes_are_named() {
        // (body, code, the question named, what the message says)
        let q = Some("q");
        let cases = [
            (
                json!({"questions": {"q": {"type": "noul", "instructions": "?"}}}),
                "invalid_request",
                None,
                "\"state\" is required",
            ),
            (
                json!({"state": "x", "questions": {}}),
                "invalid_request",
                None,
                "at least one question",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "maybe", "instructions": "?"}}}),
                "invalid_question",
                q,
                "unknown question type",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "noul"}}}),
                "invalid_question",
                q,
                "\"instructions\" is required",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}}}),
                "invalid_question",
                q,
                "needs \"criteria\": an object",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "score", "instructions": "?", "criteria": {"a": "b"}}}}),
                "invalid_question",
                q,
                "needs \"criteria\": an array",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["only one"]}}}),
                "invalid_question",
                q,
                "2 to 10 levels",
            ),
            (
                json!({"state": "x", "questions": {"q": "Is it urgent?"}}),
                "invalid_question",
                q,
                "must be an object",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": 1, "instructions": "?"}}}),
                "invalid_question",
                q,
                "\"type\" must be",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}, "eullm": {"calibration": "platt"}}),
                "invalid_request",
                None,
                "unknown calibration",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}, "eullm": {"mode": "fast"}}),
                "invalid_request",
                None,
                "unknown mode",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}, "eullm": {"temperature": 0}}),
                "invalid_request",
                None,
                "\"temperature\"",
            ),
            (
                json!({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}, "eullm": {"temprature": 2}}),
                "invalid_request",
                None,
                "unknown field",
            ),
        ];
        for (body, code, question, expected) in cases {
            let err = parse(body.clone()).unwrap_err();
            assert_eq!(err.status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
            assert_eq!(
                (err.code, err.question.as_deref()),
                (code, question),
                "{body}"
            );
            assert!(err.message.contains(expected), "{body}: {err:?}");
        }
        // The question at fault is named beside the message, not inside it,
        // where jev-style's clients would repeat it.
        let err =
            parse(json!({"state": "x", "questions": {"routing": {"type": "noul"}}})).unwrap_err();
        assert_eq!(err.question.as_deref(), Some("routing"));
        assert!(!err.message.contains("routing"), "{err:?}");
    }

    /// The error body jev-style's MCP server, guard and client read, key
    /// for key.
    #[test]
    fn errors_have_the_body_system_one_clients_parse() {
        let err = ApiError::invalid_question("team", "a choice question needs 2 to 255 options");
        assert_eq!(
            err.body(),
            json!({"error": {"code": "invalid_question", "question": "team",
                             "message": "a choice question needs 2 to 255 options"}})
        );
        // No question: no `question` key at all, not a null one.
        let err = ApiError::invalid_request("\"state\" is required");
        assert!(err.body()["error"].get("question").is_none());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn engine_errors_become_the_codes_jev_style_uses() {
        let ids = ["urgent".to_string(), "team".to_string()];
        let over = DecisionError::Question(
            1,
            Box::new(DecisionError::OverBudget(
                "question, options and readout need 2100 tokens; the model allows 2048 — \
                 nothing was truncated"
                    .into(),
            )),
        );
        let err = decision_error(over, &ids);
        assert_eq!(
            (err.status, err.code, err.question.as_deref()),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "input_budget_exceeded",
                Some("team")
            )
        );
        // ReflexBench retries a question that did not fit on this phrase.
        assert!(err.message.contains("nothing was truncated"), "{err:?}");

        let too_long = DecisionError::TooLong {
            needed: 9000,
            limit: 8192,
        };
        let err = decision_error(too_long, &ids);
        assert_eq!(
            (err.code, err.question.as_deref()),
            ("input_budget_exceeded", None)
        );
        assert!(err.message.contains("--decision-ctx"), "{err:?}");

        let err = decision_error(
            DecisionError::Question(0, Box::new(DecisionError::Invalid("no code".into()))),
            &ids,
        );
        assert_eq!(
            (err.code, err.question.as_deref()),
            ("invalid_question", Some("urgent"))
        );
        let err = decision_error(DecisionError::Invalid("bad state".into()), &ids);
        assert_eq!(
            (err.code, err.question.as_deref()),
            ("invalid_request", None)
        );
        let err = decision_error(DecisionError::Runtime("decode failed".into()), &ids);
        assert_eq!(
            (err.status, err.code),
            (StatusCode::INTERNAL_SERVER_ERROR, "internal_error")
        );
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
        assert_eq!(err.code, "invalid_request");
        assert!(err.message.contains("at most 64"), "{err:?}");
    }

    #[test]
    fn duplicate_question_ids_are_refused() {
        let text = r#"{ "state": "x", "questions": {
            "q": { "type": "noul", "instructions": "One?" },
            "q": { "type": "noul", "instructions": "Two?" } } }"#;
        let request: SystemOneRequest = serde_json::from_str(text).unwrap();
        let err = parse_request(request).unwrap_err();
        assert_eq!(
            (err.code, err.question.as_deref()),
            ("invalid_question", Some("q"))
        );
        assert!(err.message.contains("duplicate question id"), "{err:?}");
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
        let (answers, records) = build_answers(&parsed, &decision, ReadoutKind::Codes, 1.0);
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
        let (answers, records) = build_answers(&parsed, &decision, ReadoutKind::Codes, 1.0);
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
        let (answers, _) = build_answers(&parsed, &decision, ReadoutKind::Codes, 1.0);
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
    fn the_total_time_is_the_request_time_to_a_tenth_of_a_millisecond() {
        let timing = serde_json::to_value(ResponseTiming::new(975.714984)).unwrap();
        assert_eq!(timing, json!({ "total_ms": 975.7 }));
        assert_eq!(ResponseTiming::new(0.04).total_ms, 0.0);
    }

    #[test]
    fn the_state_hash_is_sha256_hex() {
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
