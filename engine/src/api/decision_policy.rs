//! The server's decision policy: rules the operator sets once for every
//! `/v1/systemone` request, applied by code before any model reads the
//! request.
//!
//! Code filters, the model judges. An option code can rule out for certain
//! is taken out of the question before the model sees it, never vetoed
//! after the model picked it. A veto leaves the client with a refused answer
//! and probabilities that still count the option that may not be chosen;
//! and a model shown an option that must not be taken can prefer it whatever
//! the facts say — offered a way to the food that ended in a trap, the
//! Jev-Style 0.8B took it 20 times out of 20 in the Snake example. Removed
//! first, the option is not part of the decision at all: the model reads and
//! answers the options that remain.
//!
//! # Configuration
//!
//! `EULLM_DECISION_POLICY` names a JSON file, read once at startup, from the
//! process environment first and the `.env` file second like every other
//! perimeter setting:
//!
//! ```json
//! { "version": 1, "deny_options": ["delete_*", "transfer_funds", "*_prod"] }
//! ```
//!
//! `deny_options` is matched against the option names of every `choice`
//! question: `*` stands for any run of characters, none included, and
//! everything else is literal. Case does not count, nor do spaces around an
//! option's name: a policy that denies `delete_all` also denies
//! `Delete_All`, which a model reads as the same option.
//!
//! `version` is required. A file written for a later version may hold rules
//! this engine does not know, and applying only the ones it does would
//! silently leave the others out, so it is refused. So is anything else that
//! does not parse — an unknown key, an empty pattern: someone who configured
//! a policy and finds it silently off is worse off than someone whose server
//! refused to start.

use std::path::Path;

use serde::Deserialize;

/// The policy file version this engine reads.
pub const POLICY_VERSION: u32 = 1;

/// Longest pattern accepted. Option names have no limit of their own, but a
/// pattern longer than this is not a name anyone types.
const MAX_PATTERN_LEN: usize = 1024;

/// The file as written. Unknown keys are refused, so a misspelt rule
/// (`deny_option`) stops the server instead of being ignored.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyFile {
    #[serde(default)]
    version: Option<u32>,
    #[serde(default)]
    deny_options: Vec<String>,
}

/// One `deny_options` entry.
#[derive(Debug, Clone)]
struct Pattern {
    /// As written, for the startup log.
    text: String,
    /// Lower-cased, for matching.
    chars: Vec<char>,
}

/// The policy in force: what `EULLM_DECISION_POLICY` configured, or no rules.
#[derive(Debug, Clone)]
pub struct DecisionPolicy {
    deny_options: Vec<Pattern>,
    /// Where it came from, for the startup log.
    source: String,
}

impl DecisionPolicy {
    /// No policy: every option reaches the model.
    pub fn none() -> Self {
        Self {
            deny_options: Vec::new(),
            source: "not configured (EULLM_DECISION_POLICY not set)".to_string(),
        }
    }

    /// The policy `EULLM_DECISION_POLICY` names, from the environment or
    /// else the `.env` file at `env_file`; [`Self::none`] when it names
    /// none. `Err` when it names a file that cannot be read or does not
    /// parse, which the caller treats as fatal.
    pub fn load(env_file: &Path) -> Result<Self, String> {
        let env_spec = std::env::var("EULLM_DECISION_POLICY").ok();
        let env_file_contents = std::fs::read_to_string(env_file).ok();
        let Some((path, source)) = Self::resolve(
            env_spec.as_deref(),
            env_file_contents.as_deref(),
            &env_file.display().to_string(),
        ) else {
            return Ok(Self::none());
        };
        let contents = std::fs::read_to_string(&path)
            .map_err(|e| format!("{source} names {path}, which cannot be read: {e}"))?;
        Self::parse(&contents, format!("{source}: {path}"))
            .map_err(|e| format!("{source} names {path}, which is not a valid policy: {e}"))
    }

    /// Pure resolution step behind [`Self::load`], so precedence is testable
    /// without mutating process environment variables: the policy file's
    /// path, and where that came from. A variable set to blanks is unset.
    fn resolve(
        env_spec: Option<&str>,
        env_file_contents: Option<&str>,
        env_file_label: &str,
    ) -> Option<(String, String)> {
        if let Some(path) = env_spec.map(str::trim).filter(|p| !p.is_empty()) {
            return Some((
                path.to_string(),
                "EULLM_DECISION_POLICY (environment)".to_string(),
            ));
        }
        let path = super::ip_allowlist::env_file_var(env_file_contents?, "EULLM_DECISION_POLICY")?;
        Some((path, format!("EULLM_DECISION_POLICY ({env_file_label})")))
    }

    /// A policy file's contents; `source` says where it came from.
    pub fn parse(contents: &str, source: String) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_str(contents).map_err(|e| e.to_string())?;
        // Only an object: serde would also read a struct from an array of its
        // fields in order, and `[1, ["x"]]` is not a policy anyone wrote.
        if !value.is_object() {
            return Err(format!(
                "a policy is a JSON object such as {{\"version\": {POLICY_VERSION}, \
                 \"deny_options\": [\"pattern\"]}}"
            ));
        }
        let file: PolicyFile = serde_json::from_value(value).map_err(|e| e.to_string())?;
        match file.version {
            Some(POLICY_VERSION) => {}
            None => {
                return Err(format!(
                    "\"version\" is required: {POLICY_VERSION}, the version this engine reads"
                ));
            }
            Some(other) => {
                return Err(format!(
                    "version {other} is not one this engine reads ({POLICY_VERSION}): it may \
                     hold rules this engine does not know, and applying only the others would \
                     leave those out"
                ));
            }
        }
        let deny_options = file
            .deny_options
            .iter()
            .enumerate()
            .map(|(i, pattern)| {
                let text = pattern.trim();
                if text.is_empty() {
                    return Err(format!("\"deny_options\" entry {i} is empty"));
                }
                if text.len() > MAX_PATTERN_LEN {
                    return Err(format!(
                        "\"deny_options\" entry {i} is longer than {MAX_PATTERN_LEN} bytes"
                    ));
                }
                if text.chars().any(char::is_control) {
                    return Err(format!(
                        "\"deny_options\" entry {i} contains a control character"
                    ));
                }
                Ok(Pattern {
                    text: text.to_string(),
                    chars: text.to_lowercase().chars().collect(),
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            deny_options,
            source,
        })
    }

    /// Whether the policy has no rules, and requests pass through untouched.
    pub fn is_empty(&self) -> bool {
        self.deny_options.is_empty()
    }

    /// Where the policy came from — for the startup log.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// One line saying what the policy does — for the startup log.
    pub fn describe(&self) -> String {
        if self.is_empty() {
            return "no rules".to_string();
        }
        let patterns: Vec<&str> = self.deny_options.iter().map(|p| p.text.as_str()).collect();
        format!(
            "deny_options {} — options matching them are removed from every choice \
             question before the model reads it",
            patterns.join(", ")
        )
    }

    /// Whether `option`, a `choice` option's name, is denied.
    pub fn denies(&self, option: &str) -> bool {
        if self.deny_options.is_empty() {
            return false;
        }
        let name: Vec<char> = option.trim().to_lowercase().chars().collect();
        self.deny_options
            .iter()
            .any(|pattern| wildcard_match(&pattern.chars, &name))
    }
}

/// Whether `pattern`, in which `*` stands for any run of characters (none
/// included), matches all of `text`. Linear in practice: on a mismatch it
/// backtracks only to the last `*`, which is enough because a later `*` can
/// absorb whatever an earlier one would have.
fn wildcard_match(pattern: &[char], text: &[char]) -> bool {
    let (mut p, mut t) = (0, 0);
    // The last `*` seen, and the position in `text` it is matched up to.
    let mut star: Option<(usize, usize)> = None;
    while t < text.len() {
        match pattern.get(p) {
            Some('*') => {
                star = Some((p, t));
                p += 1;
            }
            Some(&c) if c == text[t] => {
                p += 1;
                t += 1;
            }
            _ => match star {
                // Let the last `*` take one more character, and retry.
                Some((s, matched)) => {
                    p = s + 1;
                    t = matched + 1;
                    star = Some((s, matched + 1));
                }
                None => return false,
            },
        }
    }
    pattern[p..].iter().all(|&c| c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(json: &str) -> DecisionPolicy {
        DecisionPolicy::parse(json, "test".to_string()).expect("a valid policy")
    }

    #[test]
    fn the_environment_wins_over_the_env_file() {
        let file = "EULLM_DECISION_POLICY=/etc/eullm/from-file.json\n";
        let (path, source) =
            DecisionPolicy::resolve(Some(" /etc/eullm/policy.json "), Some(file), ".env").unwrap();
        assert_eq!(path, "/etc/eullm/policy.json");
        assert_eq!(source, "EULLM_DECISION_POLICY (environment)");

        let (path, source) = DecisionPolicy::resolve(None, Some(file), ".env").unwrap();
        assert_eq!(path, "/etc/eullm/from-file.json");
        assert_eq!(source, "EULLM_DECISION_POLICY (.env)");
    }

    #[test]
    fn unset_or_blank_means_no_policy() {
        assert!(DecisionPolicy::resolve(None, None, ".env").is_none());
        assert!(DecisionPolicy::resolve(Some("  "), None, ".env").is_none());
        assert!(DecisionPolicy::resolve(None, Some("EULLM_DECISION_POLICY=\n"), ".env").is_none());
        assert!(DecisionPolicy::resolve(None, Some("EULLM_API_KEYS=x\n"), ".env").is_none());
        let none = DecisionPolicy::none();
        assert!(none.is_empty());
        assert!(!none.denies("anything"));
        assert_eq!(none.describe(), "no rules");
    }

    #[test]
    fn a_policy_file_parses() {
        let p = policy(r#"{"version": 1, "deny_options": [" delete_* ", "transfer_funds"]}"#);
        assert!(!p.is_empty());
        assert_eq!(p.source(), "test");
        assert!(
            p.describe().contains("delete_*, transfer_funds"),
            "{}",
            p.describe()
        );
        // No rules is a valid policy: it changes nothing.
        assert!(policy(r#"{"version": 1}"#).is_empty());
        assert!(policy(r#"{"version": 1, "deny_options": []}"#).is_empty());
    }

    #[test]
    fn a_policy_that_cannot_be_applied_as_written_is_refused() {
        for (json, expected) in [
            ("", "EOF"),
            ("[]", "a policy is a JSON object"),
            (r#"[1, ["x"]]"#, "a policy is a JSON object"),
            (r#"{"deny_options": ["x"]}"#, "\"version\" is required"),
            (
                r#"{"version": 2, "deny_options": ["x"]}"#,
                "version 2 is not one this engine reads",
            ),
            (r#"{"version": 1, "deny_option": ["x"]}"#, "unknown field"),
            (r#"{"version": 1, "deny_options": "x"}"#, "invalid type"),
            (r#"{"version": 1, "deny_options": [3]}"#, "invalid type"),
            (
                r#"{"version": 1, "deny_options": ["  "]}"#,
                "entry 0 is empty",
            ),
            (
                r#"{"version": 1, "deny_options": ["a\nb"]}"#,
                "control character",
            ),
            ("{\"version\": \"1\"}", "invalid type"),
        ] {
            let err = DecisionPolicy::parse(json, "test".into()).unwrap_err();
            assert!(err.contains(expected), "{json}: {err}");
        }
        let long = format!(
            r#"{{"version": 1, "deny_options": ["{}"]}}"#,
            "a".repeat(1025)
        );
        let err = DecisionPolicy::parse(&long, "test".into()).unwrap_err();
        assert!(err.contains("longer than 1024"), "{err}");
    }

    #[test]
    fn a_policy_file_that_cannot_be_read_is_an_error_naming_it() {
        let path = std::env::temp_dir().join(format!(
            "eullm-policy-missing-{}.json",
            uuid::Uuid::new_v4()
        ));
        let env = std::env::temp_dir().join(format!("eullm-policy-env-{}", uuid::Uuid::new_v4()));
        std::fs::write(&env, format!("EULLM_DECISION_POLICY={}\n", path.display())).unwrap();
        // Only when the process environment does not set it, which the test
        // binary's does not.
        if std::env::var_os("EULLM_DECISION_POLICY").is_none() {
            let err = DecisionPolicy::load(&env).unwrap_err();
            assert!(err.contains(&path.display().to_string()), "{err}");
            assert!(err.contains("cannot be read"), "{err}");

            std::fs::write(&path, r#"{"version": 1, "deny_options": ["drop_*"]}"#).unwrap();
            let loaded = DecisionPolicy::load(&env).unwrap();
            assert!(loaded.denies("drop_table"));
            assert!(loaded.source().ends_with(&path.display().to_string()));
            std::fs::write(&path, r#"{"version": 1, "deny": []}"#).unwrap();
            let err = DecisionPolicy::load(&env).unwrap_err();
            assert!(err.contains("not a valid policy"), "{err}");
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&env);
    }

    #[test]
    fn exact_names_match_whole_names_only() {
        let p = policy(r#"{"version": 1, "deny_options": ["delete"]}"#);
        assert!(p.denies("delete"));
        assert!(!p.denies("delete_all"));
        assert!(!p.denies("undelete"));
        assert!(!p.denies(""));
    }

    #[test]
    fn a_star_stands_for_any_run_of_characters() {
        let p =
            policy(r#"{"version": 1, "deny_options": ["delete_*", "*_prod", "rm*rf", "a*b*c"]}"#);
        for denied in [
            "delete_",
            "delete_all",
            "delete_user_42",
            "db_prod",
            "_prod",
            "rmrf",
            "rm -rf",
            "rm --recursive -rf",
            "abc",
            "a-b-c",
            "aXbYbZc",
        ] {
            assert!(p.denies(denied), "{denied}");
        }
        for allowed in [
            "delete",
            "undelete_all",
            "db_prod_copy",
            "prod",
            "rm -r",
            "ab",
            "a-b-c-d",
            "keep",
        ] {
            assert!(!p.denies(allowed), "{allowed}");
        }
        let everything = policy(r#"{"version": 1, "deny_options": ["*"]}"#);
        assert!(everything.denies("anything") && everything.denies(""));
    }

    #[test]
    fn case_and_surrounding_spaces_do_not_count() {
        let p = policy(r#"{"version": 1, "deny_options": ["Transfer_*", "Ëxport"]}"#);
        assert!(p.denies("transfer_funds"));
        assert!(p.denies("TRANSFER_FUNDS"));
        assert!(p.denies("  transfer_funds "));
        assert!(p.denies("ëXPORT"));
        assert!(!p.denies("transfer"));
    }

    #[test]
    fn the_wildcard_matcher_backtracks_correctly() {
        let m = |p: &str, t: &str| {
            wildcard_match(
                &p.chars().collect::<Vec<_>>(),
                &t.chars().collect::<Vec<_>>(),
            )
        };
        assert!(m("*a", "aaa"));
        assert!(m("a*a", "aa"));
        assert!(!m("a*a", "a"));
        assert!(m("*ab*", "xaxab"));
        assert!(m("**", ""));
        assert!(m("*x*y*", "axbyc"));
        assert!(!m("*x*y*", "aybxc"));
        assert!(m("ab*cd", "abxcdcd"));
        assert!(!m("ab*cd", "abxcdc"));
    }
}
