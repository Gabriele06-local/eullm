"""The teacher gate's top-5 check points the right way.

`compare` promises "how often the quantized model's top-5 set match[es] the
reference" -- the reference top-1 kept in the candidate's top-5. It counted
the reverse (the candidate's top-1 inside the reference top-5), which scores
1.0 for a candidate that demoted the true token to rank 6 on every position,
so the >= 0.99 check could never fire on exactly the demotion it exists to
catch. The models here are stubbed rankings; what is exercised is the
reduction, which is pure arithmetic over logits.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_quantized_teacher.py"

VOCAB = 10


def _load():
    spec = importlib.util.spec_from_file_location("validate_quantized_teacher", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _RankingModel:
    """Logits with a fixed token ranking, cycling one ranking per position."""

    def __init__(self, orders):
        self._orders = orders
        self.device = torch.device("cpu")

    def __call__(self, ids):
        n = ids.shape[1]
        base = torch.zeros(n, VOCAB)
        for i in range(n):
            for rank, tok in enumerate(self._orders[i % len(self._orders)]):
                base[i, tok] = (VOCAB - rank) * 1.0
        return type("Out", (), {"logits": base.unsqueeze(0)})()


class _Tokenizer:
    def __call__(self, text, return_tensors=None, truncation=False, max_length=0):
        n = max(len(text.split()), 8)
        return {"input_ids": torch.zeros(1, n, dtype=torch.long)}


def _compare(cand_orders):
    mod = _load()
    ref = _RankingModel([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    cand = _RankingModel(cand_orders)
    return mod.compare(ref, cand, _Tokenizer(), ["parola " * 30], 64)


def test_top5_is_zero_when_the_true_token_is_demoted_everywhere():
    m = _compare([[1, 2, 3, 4, 5, 0, 6, 7, 8, 9]])
    assert m["top5_agreement"] == pytest.approx(0.0)
    assert m["top5_agreement"] < 0.99  # the gate's threshold would fail


def test_top5_counts_the_positions_where_the_true_token_survives():
    # 19 positions agree, 1 demotes the true token to rank 6: top-1 passes
    # its 0.95 threshold, so the top-5 check alone decides the gate.
    m = _compare([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 19
                 + [[1, 2, 3, 4, 5, 0, 6, 7, 8, 9]])
    assert m["top1_agreement"] >= 0.95
    assert m["top5_agreement"] == pytest.approx(m["top1_agreement"])
    assert m["top5_agreement"] < 0.99


def test_identical_models_agree_everywhere():
    m = _compare([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    assert m["top1_agreement"] == pytest.approx(1.0)
    assert m["top5_agreement"] == pytest.approx(1.0)
