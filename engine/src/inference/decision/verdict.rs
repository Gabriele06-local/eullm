//! Jev-Style decision models: every option read as a verdict at its own
//! slot.
//!
//! [Jev-Style](https://github.com/lawrence3699/jev-style) (Apache-2.0)
//! fine-tunes Qwen3.5 so that, after each option of a question, the token
//! `" ->"` is followed by `" yes"` or `" no"`. An option's score is
//! `logit(" yes") - logit(" no")` at its `" ->"` slot, and the probabilities
//! are `softmax(scores / T)` with a temperature fitted on held-out data and
//! released with the model. Every option is read in the same pass, so a
//! question can have up to 255 of them.
//!
//! The input is not a chat prompt: it has to be the one the model was
//! trained on, token for token — each segment tokenized on its own, text
//! never parsed for special tokens — and computed the way its published
//! scores were validated. This module renders it; `engine` computes it.

use std::path::Path;

use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use serde_json::Value;

use super::{DecisionError, Question};

/// The input protocols of the Jev-Style v3 releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Render {
    /// `macjev-render-v1`, Jev-Style-0.8B-Decision-v3: causal attention;
    /// the options listed, then `Judge each option:` and one slot per
    /// option. The question with its options and slots is capped at
    /// [`HEAD_MAX_V1`] tokens.
    V1,
    /// `macjev-render-v2-long-options`, layout `sb`,
    /// Jev-Style-2B-Decision-v3: block-causal attention over
    /// [`BLOCK_V2`]-token blocks; each option's slot right after it, or,
    /// when that does not fit one block, a numbered catalogue of the
    /// options followed by a numbered rubric of slots.
    V2,
}

/// Question, options and slots together, render v1.
pub(super) const HEAD_MAX_V1: usize = 2048;

/// Block size of render v2's block-causal attention.
pub(super) const BLOCK_V2: usize = 2048;

/// Whole input, both renders.
pub(super) const MAX_LEN: usize = 25_600;

/// The protocol a loaded verdict model is run with.
#[derive(Debug, Clone)]
pub(super) struct VerdictModel {
    pub name: String,
    pub render: Render,
    pub yes: LlamaToken,
    pub no: LlamaToken,
    pub arrow: LlamaToken,
    /// The calibration temperature the model was released with.
    pub temperature: f64,
    /// Where the protocol came from, for the load log.
    pub source: String,
    /// When the release was published, for a known one.
    pub release_date: Option<&'static str>,
}

/// A known release: what its `readout_config.json` says, for a GGUF that
/// arrives without it (`eullm pull` fetches only the weights).
struct Known {
    name: &'static str,
    render: Render,
    temperature: f64,
    /// Its publication date, as jev-style's release table gives it
    /// (`jev_style.models.RELEASES`): what `GET /v1/models` reports.
    release_date: &'static str,
}

/// `temperatures.global` of each release's `readout_config.json`.
const KNOWN: [Known; 2] = [
    Known {
        name: "Jev-Style-0.8B-Decision-v3",
        render: Render::V1,
        temperature: 0.880_054_682_178_933_2,
        release_date: "2026-09-24",
    },
    Known {
        name: "Jev-Style-2B-Decision-v3",
        render: Render::V2,
        temperature: 0.827_865_062_094_286_7,
        release_date: "2026-09-27",
    },
];

/// The slot tokens of every v3 release (Qwen3.5's tokenizer).
const YES: i32 = 9542;
const NO: i32 = 874;
const ARROW: i32 = 1411;

impl VerdictModel {
    /// The verdict protocol of the model at `path`, if it has one: from the
    /// `readout_config.json` next to the GGUF when there is one for this
    /// model, else from the model's name when it is a known release. The
    /// slot tokens are checked against the model's own tokenizer either way.
    ///
    /// A file of that name that is not a verdict readout, or describes
    /// another model, is left alone: it may belong to anything else kept in
    /// the same directory. One that describes this model and cannot be
    /// followed is an error — the model would otherwise be run with a
    /// protocol it was not trained on.
    pub fn detect(model: &LlamaModel, path: &Path) -> Result<Option<Self>, String> {
        let name = model.meta_val_str("general.name").unwrap_or_default();
        let config_path = path.with_file_name("readout_config.json");
        let config = config_path
            .is_file()
            .then(|| {
                std::fs::read_to_string(&config_path)
                    .map_err(|e| e.to_string())
                    .and_then(|text| {
                        serde_json::from_str::<Value>(&text).map_err(|e| e.to_string())
                    })
                    .inspect_err(|e| {
                        tracing::warn!("{}: not used, {e}", config_path.display());
                    })
                    .ok()
            })
            .flatten();
        let from_file = match config {
            Some(config) => Self::from_file(&config, &name, &config_path)?,
            None => None,
        };
        let known = KNOWN.iter().find(|k| k.name == name);
        let Some(mut verdict) = from_file.or_else(|| {
            known.map(|k| Self {
                name: name.clone(),
                render: k.render,
                yes: LlamaToken(YES),
                no: LlamaToken(NO),
                arrow: LlamaToken(ARROW),
                temperature: k.temperature,
                source: "built-in, from the release's readout_config.json".to_string(),
                release_date: None,
            })
        }) else {
            return Ok(None);
        };
        verdict.release_date = known.map(|k| k.release_date);
        for (text, want) in [
            (" yes", verdict.yes),
            (" no", verdict.no),
            (" ->", verdict.arrow),
        ] {
            let got = model
                .str_to_token_plain(text, AddBos::Never)
                .map_err(|e| e.to_string())?;
            if got != [want] {
                return Err(format!(
                    "{}: {text:?} is not the single token {} for this model's tokenizer \
                     (got {got:?}) — not the tokenizer the model was released with",
                    verdict.name, want.0
                ));
            }
        }
        Ok(Some(verdict))
    }

    /// What a `readout_config.json` next to the model says: nothing when it
    /// is not a verdict readout or describes another model.
    fn from_file(config: &Value, name: &str, path: &Path) -> Result<Option<Self>, String> {
        if config.get("readout").and_then(Value::as_str) != Some("verdict") {
            return Ok(None);
        }
        match config.get("model_name").and_then(Value::as_str) {
            Some(other) if other != name => {
                tracing::warn!(
                    "{} describes {other}, not this model ({name}): not used",
                    path.display()
                );
                Ok(None)
            }
            _ => Self::from_config(config, name, path).map(Some),
        }
    }

    fn from_config(config: &Value, name: &str, path: &Path) -> Result<Self, String> {
        let at = |pointer: &str| config.pointer(pointer);
        let text = |pointer: &str| at(pointer).and_then(Value::as_str).unwrap_or_default();
        let fail = |what: &str| format!("{}: {what}", path.display());
        if text("/readout") != "verdict" {
            return Err(fail("\"readout\" is not \"verdict\""));
        }
        let render = match (text("/format"), text("/template")) {
            ("macjev-readout-v1", "macjev-render-v1") => Render::V1,
            ("macjev-readout-v2", "macjev-render-v2-long-options")
                if text("/layout") == "sb"
                    && at("/block_size").and_then(Value::as_u64) == Some(BLOCK_V2 as u64) =>
            {
                Render::V2
            }
            (format, template) => {
                return Err(fail(&format!(
                    "unsupported readout {format:?} / template {template:?}"
                )));
            }
        };
        let token = |pointer: &str| -> Result<LlamaToken, String> {
            at(pointer)
                .and_then(Value::as_i64)
                .and_then(|id| i32::try_from(id).ok())
                .map(LlamaToken)
                .ok_or_else(|| fail(&format!("{pointer} is missing")))
        };
        let temperature = at("/temperatures/global")
            .and_then(Value::as_f64)
            .filter(|t| t.is_finite() && *t > 0.0)
            .ok_or_else(|| fail("temperatures.global is missing or not positive"))?;
        Ok(Self {
            name: name.to_string(),
            render,
            yes: token("/slot_tokens/yes/id")?,
            no: token("/slot_tokens/no/id")?,
            arrow: token("/slot_tokens/verdict_slot/id")?,
            temperature,
            source: path.display().to_string(),
            release_date: None,
        })
    }

    /// The most tokens one question with its options and slots may take:
    /// render v1's own budget; `None` for render v2, where only the whole
    /// input has one.
    pub fn question_budget(&self) -> Option<usize> {
        match self.render {
            Render::V1 => Some(HEAD_MAX_V1),
            Render::V2 => None,
        }
    }

    /// One line for the load log.
    pub fn describe(&self) -> String {
        let render = match self.render {
            Render::V1 => "render v1, causal",
            Render::V2 => "render v2, block-causal",
        };
        format!(
            "{} verdict model ({render}), calibration temperature {:.4} ({})",
            self.name, self.temperature, self.source
        )
    }
}

/// One question rendered for a verdict model.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Rendered {
    pub ids: Vec<LlamaToken>,
    /// Tokens of the state part, `State:\n<state>\n\n`.
    pub prefix_len: usize,
    /// One read position per option, in the model's option order —
    /// `false` before `true` for a yes/no question.
    pub slots: Vec<usize>,
    /// Render v2: the blocks `[a, b)` tiling `ids`, the state's first.
    pub blocks: Vec<(usize, usize)>,
}

/// The state part, `State:\n<state>\n\n`, each segment tokenized on its own.
pub(super) fn prefix<F>(encode: &F, state: &str) -> Result<Vec<LlamaToken>, DecisionError>
where
    F: Fn(&str) -> Result<Vec<LlamaToken>, DecisionError>,
{
    let mut ids = encode("State:\n")?;
    ids.extend(encode(state)?);
    ids.extend(encode("\n\n")?);
    Ok(ids)
}

/// The options as the model reads them, in its order.
fn option_texts(question: &Question) -> Vec<String> {
    match question {
        Question::Choice { options, .. } => options
            .iter()
            .map(|(name, description)| {
                if description.is_empty() {
                    name.clone()
                } else {
                    format!("{name}: {description}")
                }
            })
            .collect(),
        Question::Score { levels, .. } => levels
            .iter()
            .enumerate()
            .map(|(i, level)| format!("level {i}: {level}"))
            .collect(),
        Question::Noul {
            true_means,
            false_means,
            ..
        } => {
            let or = |means: &str, default: &str| {
                if means.is_empty() {
                    default.to_string()
                } else {
                    means.to_string()
                }
            };
            vec![
                format!(
                    "false: {}",
                    or(false_means, "no, the statement does not hold")
                ),
                format!("true: {}", or(true_means, "yes, the statement holds")),
            ]
        }
    }
}

/// `[start, stop)` cut into blocks of `BLOCK_V2` tokens.
fn blocks(start: usize, stop: usize) -> Vec<(usize, usize)> {
    (start..stop)
        .step_by(BLOCK_V2)
        .map(|a| (a, (a + BLOCK_V2).min(stop)))
        .collect()
}

/// Render `question` after the state part `prefix`.
pub(super) fn render<F>(
    verdict: &VerdictModel,
    encode: &F,
    prefix: &[LlamaToken],
    question: &Question,
) -> Result<Rendered, DecisionError>
where
    F: Fn(&str) -> Result<Vec<LlamaToken>, DecisionError>,
{
    let options = option_texts(question)
        .iter()
        .map(|o| encode(o))
        .collect::<Result<Vec<_>, _>>()?;
    let head = encode(&format!(
        "Question [{}]: {}\nOptions:\n",
        question.kind().as_str(),
        question.instructions()
    ))?;
    let newline = encode("\n")?;
    let dash = encode("- ")?;
    let arrow = verdict.arrow;
    let p = prefix.len();

    let (mut ids, slots, block_list) = match verdict.render {
        Render::V1 => {
            let mut suffix = head;
            for option in &options {
                suffix.extend(&dash);
                suffix.extend(option);
                suffix.extend(&newline);
            }
            suffix.extend(encode("Judge each option:\n")?);
            let mut slots = Vec::with_capacity(options.len());
            for option in &options {
                suffix.extend(option);
                suffix.push(arrow);
                slots.push(p + suffix.len() - 1);
                suffix.extend(&newline);
            }
            if suffix.len() > HEAD_MAX_V1 {
                return Err(DecisionError::OverBudget(format!(
                    "question, options and readout need {} tokens; {} allows {HEAD_MAX_V1} — \
                     nothing was truncated: shorten the question or the options, or split the \
                     options over several questions",
                    suffix.len(),
                    verdict.name
                )));
            }
            (suffix, slots, Vec::new())
        }
        Render::V2 => {
            let mut short = head.clone();
            let mut slots = Vec::with_capacity(options.len());
            for option in &options {
                short.extend(&dash);
                short.extend(option);
                short.push(arrow);
                slots.push(p + short.len() - 1);
                short.extend(&newline);
            }
            if short.len() <= BLOCK_V2 {
                let end = p + short.len();
                let mut b = blocks(0, p);
                b.push((p, end));
                (short, slots, b)
            } else {
                // Too long for one block: a catalogue of the options, then
                // a rubric of numbered slots.
                let mut suffix = head;
                for (i, option) in options.iter().enumerate() {
                    suffix.extend(encode(&format!("Option {}: ", i + 1))?);
                    suffix.extend(option);
                    suffix.extend(&newline);
                }
                let doc_end = p + suffix.len();
                suffix.extend(encode(
                    "Judge each numbered option in the complete catalogue above:\n",
                )?);
                let mut slots = Vec::with_capacity(options.len());
                for i in 0..options.len() {
                    suffix.extend(encode(&format!("Option {}", i + 1))?);
                    suffix.push(arrow);
                    slots.push(p + suffix.len() - 1);
                    suffix.extend(&newline);
                }
                let end = p + suffix.len();
                let mut b = blocks(0, p);
                b.extend(blocks(p, doc_end));
                b.extend(blocks(doc_end, end));
                (suffix, slots, b)
            }
        }
    };
    let total = p + ids.len();
    if total > MAX_LEN {
        return Err(DecisionError::OverBudget(format!(
            "the input needs {total} tokens (state {p} + question and options {}); {} allows \
             {MAX_LEN} — nothing was truncated: shorten the state, the question or the options",
            ids.len(),
            verdict.name
        )));
    }
    let mut all = Vec::with_capacity(total);
    all.extend_from_slice(prefix);
    all.append(&mut ids);
    Ok(Rendered {
        ids: all,
        prefix_len: p,
        slots,
        blocks: block_list,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per character, so the layout can be read off the ids.
    fn encode(text: &str) -> Result<Vec<LlamaToken>, DecisionError> {
        Ok(text.chars().map(|c| LlamaToken(c as i32)).collect())
    }

    fn text(ids: &[LlamaToken]) -> String {
        ids.iter()
            .map(|t| {
                if t.0 == ARROW {
                    "→".to_string()
                } else {
                    char::from_u32(t.0 as u32).unwrap().to_string()
                }
            })
            .collect()
    }

    fn model(render: Render) -> VerdictModel {
        VerdictModel {
            name: "test".into(),
            render,
            yes: LlamaToken(YES),
            no: LlamaToken(NO),
            arrow: LlamaToken(ARROW),
            temperature: 0.9,
            source: "test".into(),
            release_date: None,
        }
    }

    fn choice() -> Question {
        Question::Choice {
            instructions: "Which team?".into(),
            options: vec![
                ("billing".into(), "Charges".into()),
                ("tech".into(), String::new()),
            ],
        }
    }

    #[test]
    fn render_v1_lists_the_options_then_judges_each() {
        let v = model(Render::V1);
        let prefix = prefix(&encode, "Two charges.").unwrap();
        let r = render(&v, &encode, &prefix, &choice()).unwrap();
        assert_eq!(
            text(&r.ids),
            "State:\nTwo charges.\n\nQuestion [choice]: Which team?\nOptions:\n\
             - billing: Charges\n- tech\nJudge each option:\nbilling: Charges→\ntech→\n"
        );
        assert_eq!(r.prefix_len, prefix.len());
        assert!(r.slots.iter().all(|&s| r.ids[s].0 == ARROW));
        assert_eq!(r.slots.len(), 2);
        assert!(r.blocks.is_empty());
    }

    #[test]
    fn render_v2_puts_each_slot_after_its_option_in_one_block() {
        let v = model(Render::V2);
        let prefix = prefix(&encode, "Two charges.").unwrap();
        let r = render(&v, &encode, &prefix, &choice()).unwrap();
        assert_eq!(
            text(&r.ids),
            "State:\nTwo charges.\n\nQuestion [choice]: Which team?\nOptions:\n\
             - billing: Charges→\n- tech→\n"
        );
        assert!(r.slots.iter().all(|&s| r.ids[s].0 == ARROW));
        assert_eq!(
            r.blocks,
            vec![(0, r.prefix_len), (r.prefix_len, r.ids.len())]
        );
    }

    #[test]
    fn render_v2_moves_options_that_do_not_fit_one_block_to_a_catalogue() {
        let v = model(Render::V2);
        let prefix = prefix(&encode, "x").unwrap();
        let long = "d".repeat(300);
        let q = Question::Choice {
            instructions: "Which?".into(),
            options: (0..10).map(|i| (format!("o{i}"), long.clone())).collect(),
        };
        let r = render(&v, &encode, &prefix, &q).unwrap();
        let t = text(&r.ids);
        assert!(t.contains("Option 1: o0: ddd"), "{}", &t[..200]);
        assert!(t.contains(
            "Judge each numbered option in the complete catalogue above:\nOption 1→\nOption 2→\n"
        ));
        assert!(t.ends_with("Option 10→\n"));
        assert!(r.slots.iter().all(|&s| r.ids[s].0 == ARROW));
        // Blocks tile the input, none longer than a block, the rubric's own.
        assert_eq!(r.blocks.first().unwrap().0, 0);
        assert_eq!(r.blocks.last().unwrap().1, r.ids.len());
        assert!(r.blocks.windows(2).all(|w| w[0].1 == w[1].0));
        assert!(r.blocks.iter().all(|(a, b)| b - a <= BLOCK_V2));
        let rubric = t.find("Judge each").unwrap();
        assert!(r.blocks.iter().any(|&(a, _)| a == rubric));
    }

    #[test]
    fn a_yes_no_question_reads_false_before_true_with_its_meanings() {
        let v = model(Render::V1);
        let q = Question::Noul {
            instructions: "Satisfied?".into(),
            true_means: "says so".into(),
            false_means: String::new(),
        };
        let r = render(&v, &encode, &prefix(&encode, "s").unwrap(), &q).unwrap();
        let t = text(&r.ids);
        assert!(t.contains("- false: no, the statement does not hold\n- true: says so\n"));
        assert!(t.ends_with("false: no, the statement does not hold→\ntrue: says so→\n"));
    }

    #[test]
    fn a_score_question_numbers_its_levels() {
        let v = model(Render::V1);
        let q = Question::Score {
            instructions: "How urgent?".into(),
            levels: vec!["Low".into(), "High".into()],
        };
        let r = render(&v, &encode, &prefix(&encode, "s").unwrap(), &q).unwrap();
        assert!(text(&r.ids).contains("Question [score]: How urgent?\nOptions:\n- level 0: Low\n"));
    }

    #[test]
    fn budgets_refuse_rather_than_truncate() {
        let q = Question::Choice {
            instructions: "Which?".into(),
            options: (0..30).map(|i| (format!("o{i}"), "d".repeat(80))).collect(),
        };
        let small = prefix(&encode, "s").unwrap();
        let err = render(&model(Render::V1), &encode, &small, &q).unwrap_err();
        assert!(matches!(err, DecisionError::OverBudget(_)), "{err:?}");
        assert!(err.to_string().contains("nothing was truncated"), "{err}");
        let huge = prefix(&encode, &"s".repeat(MAX_LEN)).unwrap();
        let err = render(&model(Render::V2), &encode, &huge, &choice()).unwrap_err();
        assert!(matches!(err, DecisionError::OverBudget(_)), "{err:?}");
        assert!(err.to_string().contains("shorten the state"), "{err}");
    }

    #[test]
    fn a_readout_config_names_the_protocol_and_its_temperature() {
        let config: Value = serde_json::from_str(
            r#"{"format": "macjev-readout-v2", "template": "macjev-render-v2-long-options",
                "layout": "sb", "block_size": 2048, "readout": "verdict",
                "slot_tokens": {"yes": {"id": 9542}, "no": {"id": 874}, "verdict_slot": {"id": 1411}},
                "temperatures": {"global": 0.83}}"#,
        )
        .unwrap();
        let v = VerdictModel::from_config(&config, "m", Path::new("readout_config.json")).unwrap();
        assert_eq!(v.render, Render::V2);
        assert_eq!((v.yes.0, v.no.0, v.arrow.0), (YES, NO, ARROW));
        assert_eq!(v.temperature, 0.83);

        let mut other = config.clone();
        other["format"] = Value::from("macjev-readout-v9");
        assert!(VerdictModel::from_config(&other, "m", Path::new("c.json")).is_err());
        let mut cold = config;
        cold["temperatures"]["global"] = Value::from(0.0);
        assert!(VerdictModel::from_config(&cold, "m", Path::new("c.json")).is_err());
    }

    #[test]
    fn a_readout_config_of_something_else_is_left_alone() {
        let path = Path::new("readout_config.json");
        let unrelated = serde_json::json!({"readout": "linear-probe", "labels": 3});
        assert!(
            VerdictModel::from_file(&unrelated, "m", path)
                .unwrap()
                .is_none()
        );
        let another: Value = serde_json::json!({"readout": "verdict", "model_name": "other",
                                                 "format": "macjev-readout-v9"});
        assert!(
            VerdictModel::from_file(&another, "m", path)
                .unwrap()
                .is_none()
        );
        // One for this model that cannot be followed refuses the load.
        let mut ours = another;
        ours["model_name"] = Value::from("m");
        assert!(VerdictModel::from_file(&ours, "m", path).is_err());
    }
}
