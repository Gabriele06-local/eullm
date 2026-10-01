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
//! A trace that cannot be written never fails the decision it records: the
//! decision is made and audited, the client gets its answers, and the
//! server log says the trace is missing. A directory that cannot be written
//! at startup is another matter: whoever set the variable asked for the
//! traces, and a server that ran without them would leave a hole that is
//! found only when the training data is.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use serde::Serialize;

/// The file every decision's line goes to.
pub const DECISIONS_FILE: &str = "decisions.jsonl";

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

    /// Create the directory if needed and open the file for appending, so a
    /// destination that cannot be written stops the server at startup
    /// instead of losing every trace after it.
    pub fn check_writable(&self) -> Result<(), String> {
        fs::create_dir_all(&self.dir).map_err(|e| {
            format!(
                "cannot create the decision traces directory {}: {e}",
                self.dir.display()
            )
        })?;
        let path = self.decisions_path();
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map(|_| ())
            .map_err(|e| format!("cannot write {}: {e}", path.display()))
    }

    /// Append one decision's line.
    pub fn append_decision(&self, line: &impl Serialize) -> Result<(), String> {
        self.append(DECISIONS_FILE, line)
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
