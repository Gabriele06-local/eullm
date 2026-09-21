"""Tests for packaging an intermediate checkpoint as a model.

The export itself needs the training stack and several gigabytes of weights,
so what is tested here is everything that happens *before* the slow part —
which is also where the damage would be done:

  * merging an adapter into the wrong base does not raise. It produces a model
    that loads, generates, and is nonsense, and nothing downstream reports it.
  * overwriting a model directory halfway leaves something that also loads,
    and is wrong in ways nothing reports either.

Both are refused up front rather than discovered on the far side of a
twenty-minute merge.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "export_checkpoint.py")


def _load():
    spec = importlib.util.spec_from_file_location("export_checkpoint", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


export = _load()


def make_adapter(root: Path, step: int, base: str | None = "Qwen/Qwen3-4B-Base") -> Path:
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    cfg = {"peft_type": "LORA", "r": 128}
    if base is not None:
        cfg["base_model_name_or_path"] = base
    (d / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


# ── which base to merge into ─────────────────────────────────────────────

def test_base_comes_from_the_adapters_own_metadata(tmp_path):
    ckpt = make_adapter(tmp_path, 8000)
    assert export.resolve_base_model(None, ckpt) == "Qwen/Qwen3-4B-Base"


def test_an_adapter_with_no_recorded_base_refuses_rather_than_guessing(tmp_path):
    """There is no safe default. Merging into "some Qwen" is the failure."""
    ckpt = make_adapter(tmp_path, 8000, base=None)
    with pytest.raises(SystemExit, match="records no base model"):
        export.resolve_base_model(None, ckpt)


def test_explicit_base_wins_but_a_mismatch_is_announced(tmp_path, capsys):
    ckpt = make_adapter(tmp_path, 8000, base="Qwen/Qwen3-4B-Base")
    got = export.resolve_base_model("Qwen/Qwen3-8B-Base", ckpt)
    assert got == "Qwen/Qwen3-8B-Base"
    assert "differs from the adapter's recorded base" in capsys.readouterr().err


def test_a_matching_explicit_base_says_nothing(tmp_path, capsys):
    ckpt = make_adapter(tmp_path, 8000, base="Qwen/Qwen3-4B-Base")
    export.resolve_base_model("Qwen/Qwen3-4B-Base", ckpt)
    assert capsys.readouterr().err == ""


def test_a_corrupt_adapter_config_is_not_a_crash(tmp_path):
    ckpt = tmp_path / "checkpoint-8000"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{not json", encoding="utf-8")
    assert export.read_adapter_base(ckpt) is None


def test_a_full_weights_checkpoint_has_no_adapter_metadata(tmp_path):
    ckpt = tmp_path / "checkpoint-8000"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    assert export.read_adapter_base(ckpt) is None


# ── step number and provenance ───────────────────────────────────────────

def test_step_is_read_from_the_directory_name(tmp_path):
    assert export.checkpoint_step(tmp_path / "checkpoint-18000") == 18000


def test_an_unnumbered_checkpoint_has_no_step(tmp_path):
    assert export.checkpoint_step(tmp_path / "merged") is None


def test_provenance_records_which_run_and_which_step(tmp_path):
    """Two exports from one run are otherwise identical-looking directories."""
    ckpt = make_adapter(tmp_path, 8000)
    got = export.describe(ckpt, "Qwen/Qwen3-4B-Base", tmp_path / "out")
    assert got["step"] == 8000
    assert got["base_model"] == "Qwen/Qwen3-4B-Base"
    assert "checkpoint-8000" in got["source_checkpoint"]
    assert "Intermediate checkpoint" in got["note"]


def test_provenance_survives_an_unnumbered_checkpoint(tmp_path):
    ckpt = tmp_path / "final"
    ckpt.mkdir()
    assert export.describe(ckpt, "b", tmp_path / "o")["step"] is None


# ── argument checks, before anything slow happens ────────────────────────

def test_a_missing_checkpoint_fails_immediately(tmp_path):
    with pytest.raises(SystemExit, match="no such checkpoint"):
        export.main(["--checkpoint", str(tmp_path / "absent"),
                     "--output", str(tmp_path / "out")])


def test_a_non_empty_output_is_refused_without_force(tmp_path):
    """Half-overwriting a model directory produces one that loads and is wrong."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"old")
    with pytest.raises(SystemExit, match="Pass --force"):
        export.check_output_dir(out, force=False)


def test_an_empty_output_directory_passes_the_guard(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    export.check_output_dir(out, force=False)        # no exception


def test_an_absent_output_directory_passes_the_guard(tmp_path):
    export.check_output_dir(tmp_path / "not-yet", force=False)


def test_force_allows_replacing_a_populated_directory(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"old")
    export.check_output_dir(out, force=True)         # no exception


def test_help_works_without_the_training_stack(capsys):
    """torch is imported inside main, so --help works on a login node."""
    with pytest.raises(SystemExit) as e:
        export.parse_args(["--help"])
    assert e.value.code == 0
    assert "checkpoint" in capsys.readouterr().out


def test_publish_replaces_without_leaving_previous_files(tmp_path):
    """A --force publish must leave no file from the previous export behind.

    Stale shards beside the new index load without complaint and are wrong
    in ways nothing downstream reports, so the swap that replaces the old
    directory is pinned here rather than trusted to stay correct.
    """
    out = tmp_path / "merged"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"old")
    (out / "stale-extra.safetensors").write_bytes(b"stale")
    staging = tmp_path / "merged.partial"
    staging.mkdir()
    (staging / "model.safetensors").write_bytes(b"new")
    (staging / "eullm_export.json").write_bytes(b"{}")
    export.publish_staging(staging, out)
    assert sorted(p.name for p in out.iterdir()) == [
        "eullm_export.json",
        "model.safetensors",
    ]
    assert (out / "model.safetensors").read_bytes() == b"new"
    assert not staging.exists()
