//! Opt-in, local decision traces: the text of every decision, personal data
//! redacted, kept so that a decision model can later be trained on the
//! decisions it is actually asked to make (Forge reads these files).
//!
//! The audit trail records a decision's state only as a SHA-256, on purpose:
//! it is kept for every decision, and the state is the part most likely to
//! hold personal data. Keeping the text to train on is a different purpose
//! and has to be an explicit choice, so it is off unless
//! `EULLM_DECISION_TRACES` names a directory — from the process environment
//! first and the `.env` file second, like every other perimeter setting.
//! Nothing in it leaves the machine.
//!
//! `decisions.jsonl` there gets one line for every decision the audit trail
//! records, with the audit record's `id`, and the state, the questions and
//! the answers as text, with e-mail addresses, phone numbers, IBANs, codici
//! fiscali, card numbers and IPv4 addresses replaced by placeholders
//! (`audit::redact`). The line's shape is a contract with the code that
//! reads it, documented in docs/engine.md and versioned by its `schema`.
//!
//! `feedback.jsonl` gets what `POST /v1/systemone/feedback` is told about a
//! decision afterwards, under the same `id`: the answers that would have
//! been right, what came of it, and who says so. Training on a model's own
//! answers teaches it nothing it did not know; those corrections are what
//! does. Feedback is stored next to the traces, so with traces off there is
//! nowhere to put it, and the endpoint says so with a 409.
//!
//! A trace that cannot be written never fails the decision it records: the
//! decision is made and audited, the client gets its answers, and the
//! server log says the trace is missing. A directory that cannot be written
//! at startup is another matter: whoever set the variable asked for the
//! traces, and a server that ran without them would leave a hole that is
//! found only when the training data is.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use super::AppState;
use super::systemone::{ApiError, OrderedJson, OrderedMap, rejection};
use crate::audit::redact::redact;
use crate::inference::decision::{MAX_QUESTIONS, MAX_SCORE_LEVELS};

/// The file every decision's line goes to.
pub const DECISIONS_FILE: &str = "decisions.jsonl";

/// The file feedback on those decisions goes to.
pub const FEEDBACK_FILE: &str = "feedback.jsonl";

/// Longest body `/v1/systemone/feedback` reads. The most a valid feedback
/// holds — 64 answers with names of the longest accepted, and the longest
/// outcome — is about 150 KB.
pub const FEEDBACK_MAX_BODY_BYTES: usize = 256 * 1024;

/// Longest question id, and option name as an answer, a feedback may give.
const MAX_FEEDBACK_NAME_BYTES: usize = 1024;

/// Longest `outcome`.
const MAX_FEEDBACK_OUTCOME_BYTES: usize = 16 * 1024;

/// Who may be said to have given a feedback: a person, a rule in code, or a
/// teacher model.
const FEEDBACK_SOURCES: [&str; 3] = ["user", "rule", "teacher"];

/// The version of the lines' shape, written on every line. A change that
/// would break a reader of the current shape gets a new number.
pub const TRACE_SCHEMA: u32 = 1;

/// Where traces go, when they are on.
#[derive(Debug)]
pub struct DecisionTraces {
    dir: PathBuf,
    /// Where the setting came from, for the startup log.
    source: String,
    /// Held while a line is written, so that two decisions finishing at once
    /// never interleave theirs, however long the lines.
    write: Mutex<()>,
}

impl DecisionTraces {
    /// Traces at `dir`; `source` says where that came from.
    pub fn at(dir: PathBuf, source: String) -> Self {
        Self {
            dir,
            source,
            write: Mutex::new(()),
        }
    }

    /// The traces `EULLM_DECISION_TRACES` asks for, from the environment or
    /// else the `.env` file at `env_file`, checked writable: `None` when it
    /// is not set, `Err` when it names a directory that cannot be written,
    /// which the caller treats as fatal.
    pub fn load(env_file: &Path) -> Result<Option<Self>, String> {
        let env_spec = std::env::var("EULLM_DECISION_TRACES").ok();
        let env_file_contents = std::fs::read_to_string(env_file).ok();
        let Some(traces) = Self::resolve(
            env_spec.as_deref(),
            env_file_contents.as_deref(),
            &env_file.display().to_string(),
        ) else {
            return Ok(None);
        };
        traces.check_writable()?;
        Ok(Some(traces))
    }

    /// Pure resolution step behind [`Self::load`], so precedence is testable
    /// without mutating process environment variables. A variable set to
    /// blanks is unset.
    fn resolve(
        env_spec: Option<&str>,
        env_file_contents: Option<&str>,
        env_file_label: &str,
    ) -> Option<Self> {
        if let Some(dir) = env_spec.map(str::trim).filter(|d| !d.is_empty()) {
            return Some(Self::at(
                PathBuf::from(dir),
                "EULLM_DECISION_TRACES (environment)".to_string(),
            ));
        }
        let dir = super::ip_allowlist::env_file_var(env_file_contents?, "EULLM_DECISION_TRACES")?;
        Some(Self::at(
            PathBuf::from(dir),
            format!("EULLM_DECISION_TRACES ({env_file_label})"),
        ))
    }

    /// Where the setting came from — for the startup log.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The file decisions are traced to.
    pub fn decisions_path(&self) -> PathBuf {
        self.dir.join(DECISIONS_FILE)
    }

    /// The file feedback goes to.
    pub fn feedback_path(&self) -> PathBuf {
        self.dir.join(FEEDBACK_FILE)
    }

    /// Create the directory if needed and open both files for appending, so
    /// a destination that cannot be written stops the server at startup
    /// instead of losing every trace after it.
    pub fn check_writable(&self) -> Result<(), String> {
        fs::create_dir_all(&self.dir).map_err(|e| {
            format!(
                "cannot create the decision traces directory {}: {e}",
                self.dir.display()
            )
        })?;
        for path in [self.decisions_path(), self.feedback_path()] {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        Ok(())
    }

    /// Append one decision's line.
    pub fn append_decision(&self, line: &impl Serialize) -> Result<(), String> {
        self.append(DECISIONS_FILE, line)
    }

    /// Append one feedback's line.
    fn append_feedback(&self, line: &impl Serialize) -> Result<(), String> {
        self.append(FEEDBACK_FILE, line)
    }

    /// Append `line` to `file`, as one write of the line and its newline.
    fn append(&self, file: &str, line: &impl Serialize) -> Result<(), String> {
        let mut text = serde_json::to_string(line).map_err(|e| e.to_string())?;
        text.push('\n');
        let path = self.dir.join(file);
        let _one_at_a_time = self.write.lock();
        // Created again if it was removed while the server ran, as when a
        // training run moves the file away to read it.
        fs::create_dir_all(&self.dir)
            .and_then(|()| OpenOptions::new().create(true).append(true).open(&path))
            .and_then(|mut f| f.write_all(text.as_bytes()))
            .map_err(|e| format!("cannot write {}: {e}", path.display()))
    }
}

// ── Feedback ────────────────────────────────────────────────────────────

/// `POST /v1/systemone/feedback`'s body, each field read as any JSON value
/// so a wrong one is named in the error rather than in serde's words.
/// Unknown keys are refused: a misspelt `answer` must not store a feedback
/// without its answers.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FeedbackRequest {
    #[serde(default)]
    id: Option<OrderedJson>,
    #[serde(default)]
    answers: Option<OrderedJson>,
    #[serde(default)]
    outcome: Option<OrderedJson>,
    #[serde(default)]
    source: Option<OrderedJson>,
}

/// The right answer to one question, as the feedback gives it and as the
/// line keeps it: what the question's type answers with.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
enum FeedbackAnswer {
    /// `noul`: whether the statement is true.
    Noul(bool),
    /// `score`: the level's index, from 0.
    Score(u64),
    /// `choice`: the option's name.
    Choice(String),
}

/// A feedback, checked.
#[derive(Debug)]
struct Feedback {
    id: Uuid,
    answers: OrderedMap<FeedbackAnswer>,
    outcome: Option<String>,
    source: Option<&'static str>,
}

/// One line of `feedback.jsonl`. Read by Forge next to `decisions.jsonl`
/// and documented in docs/engine.md, like the decision's line.
#[derive(Debug, Serialize)]
struct FeedbackLine<'a> {
    schema: u32,
    /// `feedback`, so a line says what it is wherever it ends up.
    kind: &'static str,
    /// When the feedback was received.
    timestamp: DateTime<Utc>,
    /// The decision's audit record, as the response gave it in
    /// `eullm.audit_id`.
    id: Uuid,
    answers: &'a OrderedMap<FeedbackAnswer>,
    /// Redacted, like every text in the traces.
    outcome: Option<String>,
    source: Option<&'static str>,
}

impl Feedback {
    /// Check a request's fields: their types, their sizes, and that it says
    /// something. Whether `id` names a decision that was traced, and whether
    /// an answer is one its question could give, is for whoever joins the
    /// two files: the server keeps no index of past decisions to look them
    /// up in.
    fn read(request: FeedbackRequest) -> Result<Self, ApiError> {
        let id = match request.id {
            Some(OrderedJson::String(id)) => Uuid::parse_str(id.trim()).map_err(|_| {
                ApiError::invalid_request(format!(
                    "\"id\" must be the `eullm.audit_id` of a /v1/systemone response, a UUID \
                     such as {}; got \"{}\"",
                    Uuid::nil(),
                    id.chars().take(64).collect::<String>()
                ))
            })?,
            None => {
                return Err(ApiError::invalid_request(
                    "\"id\" is required: the `eullm.audit_id` of the /v1/systemone response \
                     the feedback is about",
                ));
            }
            Some(_) => {
                return Err(ApiError::invalid_request(
                    "\"id\" must be a string: the `eullm.audit_id` of a /v1/systemone response",
                ));
            }
        };

        let given = match request.answers {
            Some(OrderedJson::Object(answers)) => answers,
            None => {
                return Err(ApiError::invalid_request(
                    "\"answers\" is required: an object of question id → the right answer, \
                     empty when the feedback gives only an \"outcome\"",
                ));
            }
            Some(_) => {
                return Err(ApiError::invalid_request(
                    "\"answers\" must be an object of question id → the right answer",
                ));
            }
        };
        if given.0.len() > MAX_QUESTIONS {
            return Err(ApiError::invalid_request(format!(
                "at most {MAX_QUESTIONS} answers, as many as a request has questions; got {}",
                given.0.len()
            )));
        }
        let mut answers: Vec<(String, FeedbackAnswer)> = Vec::with_capacity(given.0.len());
        for (question, answer) in given.0 {
            if question.trim().is_empty() {
                return Err(ApiError::invalid_request("question ids must not be empty"));
            }
            if question.len() > MAX_FEEDBACK_NAME_BYTES {
                return Err(ApiError::invalid_request(format!(
                    "question ids are at most {MAX_FEEDBACK_NAME_BYTES} bytes"
                )));
            }
            if answers.iter().any(|(q, _)| *q == question) {
                return Err(ApiError::invalid_question(
                    &question,
                    "duplicate question id",
                ));
            }
            let answer =
                feedback_answer(answer).map_err(|e| ApiError::invalid_question(&question, e))?;
            answers.push((question, answer));
        }

        let outcome = match request.outcome {
            None => None,
            Some(OrderedJson::String(outcome)) if outcome.len() > MAX_FEEDBACK_OUTCOME_BYTES => {
                return Err(ApiError::invalid_request(format!(
                    "\"outcome\" is at most {MAX_FEEDBACK_OUTCOME_BYTES} bytes"
                )));
            }
            Some(OrderedJson::String(outcome)) => Some(outcome).filter(|o| !o.trim().is_empty()),
            Some(_) => {
                return Err(ApiError::invalid_request(
                    "\"outcome\" must be a string: what came of the decision",
                ));
            }
        };
        let source = match request.source {
            None => None,
            Some(OrderedJson::String(source)) => Some(
                FEEDBACK_SOURCES
                    .into_iter()
                    .find(|s| *s == source)
                    .ok_or_else(|| unknown_source(Some(&source)))?,
            ),
            Some(_) => return Err(unknown_source(None)),
        };
        if answers.is_empty() && outcome.is_none() {
            return Err(ApiError::invalid_request(
                "a feedback gives the right answer to at least one question, or an \"outcome\"",
            ));
        }
        Ok(Self {
            id,
            answers: OrderedMap(answers),
            outcome,
            source,
        })
    }

    /// Its line, received at `timestamp`.
    fn line(&self, timestamp: DateTime<Utc>) -> FeedbackLine<'_> {
        FeedbackLine {
            schema: TRACE_SCHEMA,
            kind: "feedback",
            timestamp,
            id: self.id,
            answers: &self.answers,
            outcome: self.outcome.as_deref().map(redact),
            source: self.source,
        }
    }
}

/// The right answer to a question as JSON gives it: an option's name for a
/// `choice`, `true` or `false` for a `noul`, a level's index for a `score`.
fn feedback_answer(answer: OrderedJson) -> Result<FeedbackAnswer, String> {
    const EXPECTED: &str = "the right answer is an option's name (choice), true or false \
                            (noul), or a level's index from 0 (score)";
    match answer {
        OrderedJson::Bool(right) => Ok(FeedbackAnswer::Noul(right)),
        OrderedJson::Number(n) => match n.as_u64() {
            Some(level) if level < MAX_SCORE_LEVELS as u64 => Ok(FeedbackAnswer::Score(level)),
            _ => Err(format!(
                "a score's level index is a whole number from 0 to {}; got {n}",
                MAX_SCORE_LEVELS - 1
            )),
        },
        OrderedJson::String(option) if option.trim().is_empty() => {
            Err("an option's name must not be empty".to_string())
        }
        OrderedJson::String(option) if option.len() > MAX_FEEDBACK_NAME_BYTES => Err(format!(
            "an option's name is at most {MAX_FEEDBACK_NAME_BYTES} bytes"
        )),
        OrderedJson::String(option) => Ok(FeedbackAnswer::Choice(option)),
        _ => Err(EXPECTED.to_string()),
    }
}

/// `source` given as something else than the three it may be; `None`: not
/// even a string.
fn unknown_source(source: Option<&str>) -> ApiError {
    let got = match source {
        Some(source) => format!("\"{}\"", source.chars().take(64).collect::<String>()),
        None => "something that is not a string".to_string(),
    };
    ApiError::invalid_request(format!(
        "\"source\" is \"user\", \"rule\" or \"teacher\", when given; got {got}"
    ))
}

/// `POST /v1/systemone/feedback`: what the right answers to a decision were,
/// appended to `feedback.jsonl` under its audit id. Behind the same
/// authentication, IP and origin checks as `/v1/systemone`, with its
/// errors.
pub(super) async fn feedback(
    State(state): State<Arc<AppState>>,
    body: Result<Json<FeedbackRequest>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Some(traces) = state.decision_traces.clone() else {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "traces_disabled",
            "decision traces are off on this server, and feedback is stored next to them: \
             start it with EULLM_DECISION_TRACES set to a directory to take feedback",
        ));
    };
    let Json(request) = body.map_err(rejection)?;
    let feedback = Feedback::read(request)?;
    let id = feedback.id;
    tokio::task::spawn_blocking(move || traces.append_feedback(&feedback.line(Utc::now())))
        .await
        .map_err(|e| ApiError::internal(format!("Feedback task failed: {e}")))?
        .map_err(|e| {
            tracing::warn!("Feedback on audit record {id} not stored: {e}");
            ApiError::internal(format!("the feedback could not be stored: {e}"))
        })?;
    Ok(Json(json!({ "id": id, "recorded": true })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("eullm-traces-{name}-{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn the_environment_wins_over_the_env_file() {
        let file = "EULLM_DECISION_TRACES=/data/from-file\n";
        let traces = DecisionTraces::resolve(Some(" /data/traces "), Some(file), ".env").unwrap();
        assert_eq!(traces.dir, PathBuf::from("/data/traces"));
        assert_eq!(traces.source(), "EULLM_DECISION_TRACES (environment)");

        let traces = DecisionTraces::resolve(None, Some(file), ".env").unwrap();
        assert_eq!(traces.dir, PathBuf::from("/data/from-file"));
        assert_eq!(traces.source(), "EULLM_DECISION_TRACES (.env)");
        assert_eq!(
            traces.decisions_path(),
            PathBuf::from("/data/from-file/decisions.jsonl")
        );
    }

    /// A feedback as a client sends it, read and checked.
    fn feedback(body: &str) -> Result<Feedback, ApiError> {
        let request: FeedbackRequest = serde_json::from_str(body)
            .map_err(|e| ApiError::invalid_request(format!("deserialize: {e}")))?;
        Feedback::read(request)
    }

    /// The `error` object of a refusal, as a client reads it.
    fn error(body: &str) -> Value {
        feedback(body).expect_err(body).body()["error"].clone()
    }

    #[test]
    fn a_feedback_is_read_in_the_order_it_was_written() {
        let read = feedback(
            r#"{ "id": " B1149E83-332B-48C0-BAB7-53725922C4DE ",
                 "answers": { "team": "billing", "is_urgent": true, "severity": 2 },
                 "outcome": "Refunded on 02/10; she wrote from mario@example.com",
                 "source": "user" }"#,
        )
        .unwrap();
        assert_eq!(read.id.to_string(), "b1149e83-332b-48c0-bab7-53725922c4de");
        assert_eq!(
            read.answers.0,
            [
                ("team".to_string(), FeedbackAnswer::Choice("billing".into())),
                ("is_urgent".to_string(), FeedbackAnswer::Noul(true)),
                ("severity".to_string(), FeedbackAnswer::Score(2)),
            ]
        );
        assert_eq!(read.source, Some("user"));

        // Its line: every key, in order, the outcome redacted.
        let timestamp = DateTime::parse_from_rfc3339("2026-10-01T14:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let line = serde_json::to_string(&read.line(timestamp)).unwrap();
        assert_eq!(
            line,
            r#"{"schema":1,"kind":"feedback","timestamp":"2026-10-01T14:00:00Z","id":"b1149e83-332b-48c0-bab7-53725922c4de","answers":{"team":"billing","is_urgent":true,"severity":2},"outcome":"Refunded on 02/10; she wrote from [EMAIL]","source":"user"}"#
        );
    }

    #[test]
    fn an_outcome_alone_is_a_feedback_and_absent_fields_are_null() {
        let read = feedback(
            r#"{ "id": "b1149e83-332b-48c0-bab7-53725922c4de", "answers": {},
                 "outcome": "The customer was satisfied", "source": null }"#,
        )
        .unwrap();
        let line = serde_json::to_value(read.line(Utc::now())).unwrap();
        assert_eq!(line["answers"], json!({}));
        assert_eq!(line["outcome"], "The customer was satisfied");
        assert_eq!(line["source"], Value::Null);

        let read = feedback(
            r#"{ "id": "b1149e83-332b-48c0-bab7-53725922c4de", "answers": { "q": false } }"#,
        )
        .unwrap();
        let line = serde_json::to_value(read.line(Utc::now())).unwrap();
        assert_eq!(line["outcome"], Value::Null);
        assert_eq!(line["source"], Value::Null);
        assert_eq!(line["kind"], "feedback");
    }

    #[test]
    fn a_feedback_that_says_nothing_or_the_wrong_thing_is_refused() {
        let id = r#""id": "b1149e83-332b-48c0-bab7-53725922c4de""#;
        // (body, code, the question named, what the message says)
        let cases = [
            (
                r#"{ "answers": { "q": true } }"#.to_string(),
                "invalid_request",
                None,
                "\"id\" is required",
            ),
            (
                r#"{ "id": 7, "answers": { "q": true } }"#.to_string(),
                "invalid_request",
                None,
                "must be a string",
            ),
            (
                r#"{ "id": "decision-7", "answers": { "q": true } }"#.to_string(),
                "invalid_request",
                None,
                "a UUID",
            ),
            (
                format!(r#"{{ {id} }}"#),
                "invalid_request",
                None,
                "\"answers\" is required",
            ),
            (
                format!(r#"{{ {id}, "answers": ["billing"] }}"#),
                "invalid_request",
                None,
                "must be an object",
            ),
            (
                format!(r#"{{ {id}, "answers": {{}} }}"#),
                "invalid_request",
                None,
                "at least one question",
            ),
            (
                format!(r#"{{ {id}, "answers": {{}}, "outcome": "  " }}"#),
                "invalid_request",
                None,
                "at least one question",
            ),
            (
                format!(r#"{{ {id}, "answers": {{ "q": null }} }}"#),
                "invalid_question",
                Some("q"),
                "option's name (choice)",
            ),
            (
                format!(r#"{{ {id}, "answers": {{ "q": ["a"] }} }}"#),
                "invalid_question",
                Some("q"),
                "option's name (choice)",
            ),
            (
                format!(r#"{{ {id}, "answers": {{ "q": 0.7 }} }}"#),
                "invalid_question",
                Some("q"),
                "whole number from 0 to 9",
            ),
            (
                format!(r#"{{ {id}, "answers": {{ "q": -1 }} }}"#),
                "invalid_question",
                Some("q"),
                "whole number from 0 to 9",
            ),
            (
                format!(r#"{{ {id}, "answers": {{ "q": 10 }} }}"#),
                "invalid_question",
                Some("q"),
                "whole number from 0 to 9",
            ),
            (
                format!(r#"{{ {id}, "answers": {{ "q": " " }} }}"#),
                "invalid_question",
                Some("q"),
                "must not be empty",
            ),
            (
                format!(r#"{{ {id}, "answers": {{ "q": true, "q": false }} }}"#),
                "invalid_question",
                Some("q"),
                "duplicate question id",
            ),
            (
                format!(r#"{{ {id}, "answers": {{ "": true }} }}"#),
                "invalid_request",
                None,
                "must not be empty",
            ),
            (
                format!(r#"{{ {id}, "answers": {{ "q": true }}, "outcome": 3 }}"#),
                "invalid_request",
                None,
                "\"outcome\" must be a string",
            ),
            (
                format!(r#"{{ {id}, "answers": {{ "q": true }}, "source": "boss" }}"#),
                "invalid_request",
                None,
                "\"user\", \"rule\" or \"teacher\"",
            ),
            (
                format!(r#"{{ {id}, "answers": {{ "q": true }}, "source": 1 }}"#),
                "invalid_request",
                None,
                "\"user\", \"rule\" or \"teacher\"",
            ),
            (
                format!(r#"{{ {id}, "answer": {{ "q": true }} }}"#),
                "invalid_request",
                None,
                "unknown field",
            ),
        ];
        for (body, code, question, expected) in cases {
            let error = error(&body);
            assert_eq!(error["code"], code, "{body}: {error}");
            assert_eq!(error["question"].as_str(), question, "{body}: {error}");
            assert!(
                error["message"].as_str().unwrap().contains(expected),
                "{body}: {error}"
            );
        }
    }

    #[test]
    fn a_feedback_too_large_to_be_one_is_refused() {
        let id = r#""id": "b1149e83-332b-48c0-bab7-53725922c4de""#;
        let many: Vec<String> = (0..=MAX_QUESTIONS)
            .map(|i| format!(r#""q{i}": true"#))
            .collect();
        let refused = error(&format!(
            r#"{{ {id}, "answers": {{ {} }} }}"#,
            many.join(", ")
        ));
        assert!(
            refused["message"]
                .as_str()
                .unwrap()
                .contains("at most 64 answers"),
            "{refused}"
        );

        let long = "x".repeat(MAX_FEEDBACK_NAME_BYTES + 1);
        let refused = error(&format!(r#"{{ {id}, "answers": {{ "{long}": true }} }}"#));
        assert!(
            refused["message"]
                .as_str()
                .unwrap()
                .contains("at most 1024 bytes"),
            "{refused}"
        );
        let refused = error(&format!(r#"{{ {id}, "answers": {{ "q": "{long}" }} }}"#));
        assert_eq!(refused["question"], "q");

        let outcome = "x".repeat(MAX_FEEDBACK_OUTCOME_BYTES + 1);
        let refused = error(&format!(
            r#"{{ {id}, "answers": {{}}, "outcome": "{outcome}" }}"#
        ));
        assert!(
            refused["message"]
                .as_str()
                .unwrap()
                .contains("at most 16384 bytes"),
            "{refused}"
        );

        // The most a valid feedback holds fits the body limit.
        let answers: Vec<String> = (0..MAX_QUESTIONS)
            .map(|i| {
                format!(
                    r#""{}{i:02}": "{}""#,
                    "q".repeat(MAX_FEEDBACK_NAME_BYTES - 2),
                    &long[1..]
                )
            })
            .collect();
        let body = format!(
            r#"{{ {id}, "answers": {{ {} }}, "outcome": "{}", "source": "teacher" }}"#,
            answers.join(", "),
            &outcome[1..]
        );
        assert!(body.len() < FEEDBACK_MAX_BODY_BYTES, "{}", body.len());
        assert!(feedback(&body).is_ok());
    }

    #[test]
    fn unset_or_blank_means_no_traces() {
        assert!(DecisionTraces::resolve(None, None, ".env").is_none());
        assert!(DecisionTraces::resolve(Some(" "), None, ".env").is_none());
        assert!(DecisionTraces::resolve(None, Some("EULLM_DECISION_TRACES=\n"), ".env").is_none());
    }

    #[test]
    fn an_unwritable_directory_is_reported() {
        let dir = scratch("ok");
        let traces = DecisionTraces::at(dir.join("nested"), "test".into());
        assert!(traces.check_writable().is_ok(), "creates the directory");
        assert!(traces.decisions_path().exists());
        let _ = fs::remove_dir_all(&dir);

        // A path whose parent is a file cannot be a directory.
        let file = scratch("bad");
        fs::write(&file, b"x").unwrap();
        let traces = DecisionTraces::at(file.join("traces"), "test".into());
        let err = traces.check_writable().unwrap_err();
        assert!(err.contains("cannot create"), "{err}");
        assert!(traces.append_decision(&serde_json::json!({})).is_err());
        let _ = fs::remove_file(&file);
    }

    /// Every line survives concurrent writers, whole and on its own line,
    /// long ones included, and the file comes back if it is moved away.
    #[test]
    fn concurrent_lines_never_interleave() {
        let dir = scratch("race");
        let traces = Arc::new(DecisionTraces::at(dir.clone(), "test".into()));
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let traces = Arc::clone(&traces);
                std::thread::spawn(move || {
                    for i in 0..25 {
                        // Lines far longer than one pipe buffer.
                        let state = "x".repeat(10_000 + 997 * i);
                        let line = serde_json::json!({ "thread": t, "i": i, "state": state });
                        traces.append_decision(&line).unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let contents = fs::read_to_string(traces.decisions_path()).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 200);
        for line in lines {
            let value: serde_json::Value = serde_json::from_str(line).expect("a whole line");
            assert!(value["state"].as_str().unwrap().len() >= 10_000);
        }

        fs::remove_file(traces.decisions_path()).unwrap();
        traces
            .append_decision(&serde_json::json!({"again": true}))
            .unwrap();
        assert_eq!(
            fs::read_to_string(traces.decisions_path()).unwrap(),
            "{\"again\":true}\n"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
