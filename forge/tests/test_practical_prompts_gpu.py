"""The practical-prompts job keeps the retrieval GPU free for the teacher.

`make_practical_prompts.py` loads Qwen3-Embedding and Qwen3-Reranker on the
first visible GPU and then starts one llama-server per --gpus entry. Starting
one on that same GPU makes llama-server fail to create the 27B teacher's
128k context beside them, the way the teacher exam died before #1045.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_practical_prompts.py"


def _load():
    spec = importlib.util.spec_from_file_location("make_practical_prompts", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Server:
    started = []

    def __init__(self, gguf, **kw):
        self.devices = kw["devices"]
        self.url = f"http://127.0.0.1/{self.devices}"
        _Server.started.append(self.devices)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Index:
    records = []


def test_teacher_servers_skip_the_gpu_the_retrieval_models_use(monkeypatch, tmp_path):
    import argparse

    mod = _load()
    ap = argparse.ArgumentParser()
    assert mod.teacher_gpus("0,1,2", True, ap) == ["1", "2"]
    assert mod.teacher_gpus("0,1,2", False, ap) == ["0", "1", "2"]
    with pytest.raises(SystemExit):
        mod.teacher_gpus("0", True, ap)

    _Server.started = []
    monkeypatch.setattr("eullm_forge.eval.dense.build_hybrid", lambda *a, **k: _Index())
    monkeypatch.setattr("eullm_forge.eval.gguf.LlamaServer", _Server)
    monkeypatch.setattr(mod, "groups", lambda *a, **k: [])
    out = tmp_path / "prompts.jsonl"
    assert mod.main(["--norms", str(tmp_path / "n.jsonl"), "--embedder", "e",
                     "--serve", "t.gguf", "--gpus", "0,1,2", "--out", str(out)]) == 1
    assert _Server.started == ["1", "2"]
