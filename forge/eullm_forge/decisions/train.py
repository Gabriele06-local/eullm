"""A decision model of your own: LoRA SFT of a small chat model on the
answer code, merged and exported to GGUF.

Each example is the engine's codes-readout prompt for one question about
one state (`prompt.CodeReadout`), and the loss is the cross-entropy of the
right answer's code token at the end of it — nothing else. The engine reads
exactly that position and nothing after it, so there is no end-of-turn to
learn and no question to imitate; whatever the base model does with the
rest of a conversation is left as it was.

The base is a small Apache-2.0 chat model whose template can switch
reasoning off, Qwen3-1.7B by default (Qwen3-0.6B for a reflex on a CPU):
the engine asks for the answer's first token right after the template's
empty reasoning block, and a model that must reason first spends that token
on its tag. Training reuses Forge's own pieces — `identity.load_text_model`
and `lora_target_modules` for the model, the HF Trainer as stage 3 runs it,
resuming from the last checkpoint, `identity.merge_identity_adapter` and
`export.export_gguf` for the GGUF — and writes beside the adapter what the
dev split said before and after, so a run reports whether it helped.

Two things differ from instruction tuning, both deliberate:

* **Only the answer position's logits are computed.** The loss needs one
  row of the vocabulary-sized logits per example, not one per token; at
  2,048 tokens and Qwen3's 151,936-entry vocabulary the full tensor would
  be 1.2 GB per example in fp32, all of it thrown away. The batch is
  padded on the left so the answer position is the last one of every row
  (`logits_to_keep=1`), with positions counted from each prompt's own
  start, as the engine counts them.
* **The default export is Q8_0, not Q4_K_M.** A decision is a probability,
  and quantization noise moves it: measured on Qwen3-0.6B on a 4-core CPU,
  the same question answered in two evaluation modes differs by up to 0.017
  at F16, 0.13 at Q8_0 and 0.34 at Q4_K_M (docs/engine.md). A 1.7B model at
  Q8_0 is under 2 GB; the qualification test measures the noise either way.

The temperature fitted on the dev split goes into the GGUF the export
writes (`metrics.TEMPERATURE_KEY`), which the engine applies by default:
the model is served calibrated without every client having to say how.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset import Example, read_examples
from .metrics import (
    TEMPERATURE_KEY,
    at_temperature,
    check_temperature,
    class_result,
    fit_temperature,
    summarize,
)
from .prompt import CodeReadout

logger = logging.getLogger(__name__)

#: Apache-2.0, a chat template with a reasoning switch, and small enough to
#: train on a 16 GB GPU in bf16 with room for long states.
DEFAULT_BASE = "Qwen/Qwen3-1.7B"
#: What `decisions export` quantizes to; see the module docstring.
DEFAULT_QUANT = "q8_0"
#: Written next to the adapter: what was trained, on what, and how it did.
REPORT = "decision-model.json"
#: `export_decision_model(temperature=FITTED)`: the temperature the run
#: fitted on its dev split, the one its probabilities are calibrated at.
FITTED = "fitted"


@dataclass
class DecisionTrainConfig:
    """LoRA SFT of a decision model.

    Attributes:
        dataset_dir: what `decisions build` wrote (train.jsonl, dev.jsonl).
        output_dir: checkpoints, the adapter and the report.
        base_model: HF id or local path of the base chat model.
        lora_rank / lora_alpha / lora_dropout: the adapter.
        num_epochs: passes over the training split. A few hundred examples
            a question learn in two or three; more starts fitting noise,
            which shows up as a dev ECE that grows while accuracy does not.
        learning_rate: AdamW's, with linear decay; stage 3's and identity's.
        batch_size / gradient_accumulation_steps: per-device batch and
            accumulation; the effective batch is their product.
        max_length: longest prompt kept, in tokens. A longer one is dropped,
            not cut: the engine refuses to truncate a state, and a model
            trained on cut states learns from text it is never shown.
        gradient_checkpointing: trade compute for activation memory.
        save_steps: checkpoint every N steps (0: once per epoch). A run
            finding a checkpoint in output_dir resumes from it.
        eval_batch_size: prompts scored at once on the dev split.
        baseline: score the base model on dev before training too.
        seed: for the shuffling and the adapter's initialization.
    """

    dataset_dir: str = ""
    output_dir: str = ""
    base_model: str = DEFAULT_BASE
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_epochs: float = 2.0
    learning_rate: float = 2e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_length: int = 2048
    gradient_checkpointing: bool = True
    save_steps: int = 0
    eval_batch_size: int = 8
    baseline: bool = True
    seed: int = 0


def build_features(examples: list[Example], readout: CodeReadout,
                   max_length: int) -> tuple[list[dict], Counter]:
    """Each example as the engine will prompt the trained model: the
    prompt's token ids, the code token the loss is put on, and every
    class's spellings for scoring. Examples the engine would not ask are
    dropped and counted with the reason."""
    features: list[dict] = []
    dropped: Counter = Counter()
    for e in examples:
        try:
            classes = readout.class_tokens(e.question)
        except ValueError as err:
            dropped[f"this model cannot be asked it: {err}"] += 1
            continue
        ids = readout.tokens(e.state, e.question)
        if ids is None:
            dropped["holds a control token's text the template cannot keep apart"] += 1
            continue
        if len(ids) > max_length:
            dropped[f"longer than {max_length} tokens"] += 1
            continue
        features.append({
            "input_ids": ids,
            "target": classes[e.label][0],
            "classes": classes,
            "label": e.label,
            "kind": e.question.kind,
        })
    return features, dropped


def collate_decisions(features: list[dict], pad_token_id: int) -> dict:
    """Pad a batch on the left, so every row's answer position is its last.

    Positions count from each prompt's own first token, as they do when the
    engine decodes it alone; the padding is masked out of attention.
    """
    import torch

    width = max(len(f["input_ids"]) for f in features)
    ids, mask, positions = [], [], []
    for f in features:
        n = len(f["input_ids"])
        pad = width - n
        ids.append([pad_token_id] * pad + f["input_ids"])
        mask.append([0] * pad + [1] * n)
        positions.append([0] * pad + list(range(n)))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
        "position_ids": torch.tensor(positions, dtype=torch.long),
        "target": torch.tensor([f["target"] for f in features], dtype=torch.long),
    }


def answer_logits(model, batch: dict):
    """The next-token logits at each row's last position, in fp32."""
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        logits_to_keep=1,
        use_cache=False,
    )
    return out.logits[:, -1, :].float()


def score_features(model, features: list[dict], pad_token_id: int,
                   batch_size: int = 8) -> list[dict]:
    """Read every prompt's answer as the engine reads it: per class, the log
    of its spellings' summed probability over the whole vocabulary."""
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    results: list[dict | None] = [None] * len(features)
    # Similar lengths together: less padding to decode.
    order = sorted(range(len(features)), key=lambda i: len(features[i]["input_ids"]))
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            batch = collate_decisions([features[i] for i in chosen], pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            logp = torch.log_softmax(answer_logits(model, batch), dim=-1)
            for row, i in zip(logp, chosen):
                f = features[i]
                lps = [torch.logsumexp(row[torch.tensor(t, device=device)], 0).item()
                       for t in f["classes"]]
                results[i] = class_result(lps, f["label"], f["kind"])
    if was_training:
        model.train()
    return results


def train_decision_model(config: DecisionTrainConfig) -> str:
    """Train; return the adapter's path. The report goes beside it."""
    dataset = Path(config.dataset_dir)
    train_path, dev_path = dataset / "train.jsonl", dataset / "dev.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"no train.jsonl in {dataset}: run `decisions build` first")
    if not config.output_dir:
        raise ValueError("output_dir is required")
    train_examples = read_examples(train_path)
    dev_examples = read_examples(dev_path) if dev_path.is_file() else []
    if not train_examples:
        raise ValueError(f"{train_path} holds no example")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    from ..identity import load_text_model, lora_target_modules

    # Before the adapter exists: its initialization is random too.
    set_seed(config.seed)
    logger.info("Decision model: %s on %s", config.base_model, dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    readout = CodeReadout(tokenizer)
    logger.info("  Prompt: %s", readout.describe())
    train_features, dropped_train = build_features(train_examples, readout, config.max_length)
    dev_features, dropped_dev = build_features(dev_examples, readout, config.max_length)
    for name, dropped in (("train", dropped_train), ("dev", dropped_dev)):
        for reason, n in dropped.items():
            logger.warning("  %s: dropped %d examples: %s", name, n, reason)
    if not train_features:
        raise ValueError("no training example is left to train on")
    logger.info("  Examples: %d train, %d dev", len(train_features), len(dev_features))

    # bf16 where the GPU has it; fp32 on a CPU, which cannot train in half
    # precision (identity.fine_tune_identity, same reasons).
    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    fp16 = use_cuda and not bf16
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    model = load_text_model(config.base_model, dtype=dtype,
                            device_map="auto" if use_cuda else None)

    report: dict = {
        "base_model": config.base_model,
        "dataset": str(dataset),
        "prompt": readout.describe(),
        "examples": {"train": len(train_features), "dev": len(dev_features)},
        "dropped": {"train": dict(dropped_train), "dev": dict(dropped_dev)},
        "config": asdict(config),
    }
    if config.baseline and dev_features:
        report["dev_before"] = summarize(
            score_features(model, dev_features, pad_token_id, config.eval_batch_size))
        logger.info("  Dev before training: %s", _line(report["dev_before"]))

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # With a frozen base the embeddings' output needs grad, or
        # checkpointed blocks have nothing to backpropagate into the LoRA.
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=lora_target_modules(model),
    ))

    class DecisionTrainer(Trainer):
        """The loss on the answer position only (see the module docstring)."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            target = inputs.pop("target")
            logits = answer_logits(model, inputs)
            loss = torch.nn.functional.cross_entropy(logits, target)
            return (loss, {"logits": logits}) if return_outputs else loss

    class Rows(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    output_dir = Path(config.output_dir)
    trainer = DecisionTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            bf16=bf16,
            fp16=fp16,
            logging_steps=10,
            save_strategy="steps" if config.save_steps else "epoch",
            save_steps=config.save_steps or 500,
            save_total_limit=2,
            report_to="none",
            seed=config.seed,
            # `target` is not an argument of the model's forward: kept.
            remove_unused_columns=False,
        ),
        train_dataset=Rows(train_features),
        data_collator=lambda rows: collate_decisions(rows, pad_token_id),
    )
    # The loss above is a mean per micro-batch: the Trainer divides it by
    # the accumulation steps only when told the model computes no loss of
    # its own.
    trainer.model_accepts_loss_kwargs = False
    last = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    if last:
        logger.info("  Resuming from %s", last)
    trainer.train(resume_from_checkpoint=last)

    if dev_features:
        results = score_features(model, dev_features, pad_token_id, config.eval_batch_size)
        report["dev_after"] = summarize(results)
        logger.info("  Dev after training:  %s", _line(report["dev_after"]))
        # Measured on the split it is fitted on, so a little flattering: the
        # qualification test, on held-out states, says whether it holds.
        temperature = fit_temperature(results)
        report["dev_temperature"] = temperature
        report["dev_after_at_temperature"] = summarize(at_temperature(results, temperature))
        logger.info("  Dev at temperature %.2f: %s", temperature,
                    _line(report["dev_after_at_temperature"]))

    adapter = output_dir / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    (output_dir / REPORT).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    logger.info("Adapter saved to %s", adapter)
    return str(adapter)


def _line(summary: dict) -> str:
    return "; ".join(
        f"{kind} n={s['n']} accuracy {s['accuracy']:.3f} ECE {s['ece']:.3f} "
        f"coverage {s['coverage']:.3f}"
        for kind, s in summary.items()
    )


def recorded_base(run_dir: str | Path) -> str | None:
    """The base the adapter in `run_dir` was trained on: the report's, else
    the adapter's own metadata — never a guess, since merging into the wrong
    base gives a model that loads and answers noise."""
    run = Path(run_dir)
    for path, key in ((run / REPORT, "base_model"),
                      (run / "adapter" / "adapter_config.json", "base_model_name_or_path")):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get(key)
        except (OSError, ValueError):
            continue
        if value:
            return value
    return None


def export_temperature(run_dir: str | Path, temperature=FITTED) -> tuple[float | None, str]:
    """The temperature the run's GGUF is to carry, as the engine will read
    it, and where it comes from: FITTED for the one the run fitted on its
    dev split (none when it had no dev split to fit it on), a number for
    that one, None for none — the engine then applies 1. Raises ValueError
    for a temperature the engine would refuse."""
    if temperature is None:
        return None, "none asked for"
    if temperature == FITTED:
        try:
            report = json.loads((Path(run_dir) / REPORT).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
        fitted = report.get("dev_temperature")
        if fitted is None:
            return None, "the run fitted none: it had no dev split"
        return check_temperature(fitted), "fitted on the dev split"
    return check_temperature(temperature), "given"


def export_decision_model(run_dir: str, output: str, quantization: str = DEFAULT_QUANT,
                          base_model: str | None = None, temperature=FITTED) -> str:
    """Merge the run's adapter into its base and export it to GGUF; return
    the GGUF's path. The merged model is written to `<run_dir>/merged`.

    `temperature` goes into the GGUF as `eullm.decision.temperature`, the
    one the engine applies by default (see `export_temperature`): the
    dev-fitted one unless given, None for none."""
    from ..export import ExportConfig, export_gguf
    from ..identity import merge_identity_adapter

    run = Path(run_dir)
    adapter = run / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"no trained adapter in {run}: run `decisions train` first")
    base = base_model or recorded_base(run)
    if not base:
        raise ValueError(f"{run} does not say which base it was trained on: pass --base")
    # Before the merge: a temperature the engine would refuse costs nothing yet.
    value, _ = export_temperature(run, temperature)
    merged = run / "merged"
    if merged.exists():
        # Stale shards beside new ones load without complaint and are wrong.
        shutil.rmtree(merged)
    merge_identity_adapter(base, str(adapter), str(merged))
    # None as well as a value: a GGUF exported without a temperature must
    # not carry one, whatever wrote it before.
    metadata = {TEMPERATURE_KEY: None if value is None else ("float32", value)}
    return export_gguf(ExportConfig(model_path=str(merged), output_path=output,
                                    quantization=quantization, metadata=metadata))
