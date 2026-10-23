#!/usr/bin/env python3
"""Fine-tune Qwen3-Reranker into the context gate: are these texts enough to answer?

    python forge/scripts/train_context_gate.py --train $WORK/eullm_runs/gate/train.jsonl \\
        --dev $WORK/eullm_runs/gate/dev.jsonl --out $WORK/eullm_runs/gate/model

The reranker already answers "yes"/"no" to "does this text answer this
question"; the gate asks it of the three texts together, with its own
instruction (`context_gate.GATE_TASK`), trained on examples whose label is
known by construction (make_gate_pairs.py). The loss is the cross-entropy of
"yes" against "no" at the last position, exactly where `dense.Reranker`
reads its score, so the trained model is used through that same class.

The development set (articles and off-topic matters never trained on) picks
the epoch to keep and the threshold, the one that best balances the two
errors; both go in ``gate.json`` beside the weights, with the development
scores per reason. A directory that already holds ``gate.json`` is not
trained again.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.datasets.context_gate import GATE_TASK  # noqa: E402
from eullm_forge.eval.dense import Reranker  # noqa: E402


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode(tok, row: dict, prefix: list[int], suffix: list[int], max_length: int) -> list[int]:
    """The ids `Reranker.scores` builds for (question, texts), the texts cut to fit."""
    head = tok.encode(f"<Instruct>: {GATE_TASK}\n<Query>: {row['question']}\n<Document>: ",
                      add_special_tokens=False)
    doc = tok.encode(row["document"], add_special_tokens=False)
    room = max(0, max_length - len(prefix) - len(suffix) - len(head))
    return prefix + head + doc[:room] + suffix


def best_threshold(probs: list[float], labels: list[int]) -> tuple[float, float]:
    """The threshold with the best balanced accuracy, and that accuracy."""
    pos = sum(labels)
    neg = len(labels) - pos
    best = (0.5, 0.0)
    for t in sorted({0.5, *[round(p, 3) for p in probs]}):
        tp = sum(1 for p, y in zip(probs, labels) if y and p >= t)
        tn = sum(1 for p, y in zip(probs, labels) if not y and p < t)
        bal = ((tp / pos if pos else 1.0) + (tn / neg if neg else 1.0)) / 2
        if bal > best[1]:
            best = (t, bal)
    return best


def capped_threshold(probs: list[float], labels: list[int], threshold: float,
                     max_false_gate: float) -> float:
    """``threshold``, lowered until at most ``max_false_gate`` of the sufficient
    examples fall below it.

    A question gated wrongly loses an answer the model had; one let through
    wrongly is still answered by a model trained to say what is missing. The
    version-1 gate, chosen on balance alone, stopped half the practical exam.
    """
    good = sorted(p for p, y in zip(probs, labels) if y)
    if not good:
        return threshold
    allowed = int(max_false_gate * len(good))
    return min(threshold, good[allowed]) if allowed < len(good) else threshold


def report(rows: list[dict], probs: list[float], threshold: float) -> dict:
    """Accuracy at the threshold, overall and per reason."""
    by = defaultdict(lambda: [0, 0])
    for r, p in zip(rows, probs):
        right = (p >= threshold) == bool(r["label"])
        by[r["reason"]][0] += right
        by[r["reason"]][1] += 1
    total = sum(v[0] for v in by.values()) / max(1, len(rows))
    return {"accuracy": round(total, 4),
            "by_reason": {k: round(v[0] / v[1], 4) for k, v in sorted(by.items())},
            "n": len(rows)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--max-false-gate", type=float, default=0.03,
                    help="at most this share of the sufficient development examples gated")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if (args.out / "gate.json").is_file():
        print(f"[gate] {args.out}/gate.json exists: already trained", flush=True)
        return 0
    import torch

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    scorer = Reranker(args.base, task=GATE_TASK, max_length=args.max_length)
    tok, model = scorer.tok, scorer.model
    model.float().train()
    dev_rows, train_rows = read(args.dev), read(args.train)
    enc = [(encode(tok, r, scorer.prefix, scorer.suffix, args.max_length), r["label"])
           for r in train_rows]
    dev_enc = [encode(tok, r, scorer.prefix, scorer.suffix, args.max_length) for r in dev_rows]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(enc) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * max(0.0, 1 - s / steps))
    device = model.device

    def logits_of(batch_ids: list[list[int]]):
        padded = tok.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            last = model(**padded).logits[:, -1, :]
        return torch.stack([last[:, scorer.no], last[:, scorer.yes]], dim=1).float()

    def dev_probs() -> list[float]:
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(dev_enc), args.batch):
                two = logits_of(dev_enc[i:i + args.batch])
                out += torch.softmax(two, dim=1)[:, 1].tolist()
        model.train()
        return out

    best = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(enc)
        batches = [enc[i:i + args.batch] for i in range(0, len(enc), args.batch)]
        for step, batch in enumerate(batches, 1):
            two = logits_of([ids for ids, _ in batch])
            target = torch.tensor([y for _, y in batch], device=device)
            loss = torch.nn.functional.cross_entropy(two, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(f"[gate] epoch {epoch} step {step}/{len(batches)} loss {loss.item():.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        probs = dev_probs()
        labels = [r["label"] for r in dev_rows]
        threshold, balanced = best_threshold(probs, labels)
        threshold = capped_threshold(probs, labels, threshold, args.max_false_gate)
        dev = report(dev_rows, probs, threshold)
        print(f"[gate] epoch {epoch}: dev balanced accuracy {balanced:.4f} at threshold "
              f"{threshold:.3f}; {json.dumps(dev['by_reason'])}", flush=True)
        if best is None or balanced > best["balanced_accuracy"]:
            best = {"epoch": epoch, "threshold": threshold, "balanced_accuracy": balanced,
                    "dev": dev}
            args.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
    meta = {"task": GATE_TASK, "base": args.base, "max_length": args.max_length,
            "train_examples": len(enc), **best}
    (args.out / "gate.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[gate] kept epoch {best['epoch']}: threshold {best['threshold']:.3f}, dev "
          f"balanced accuracy {best['balanced_accuracy']:.4f} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
