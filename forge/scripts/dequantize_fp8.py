#!/usr/bin/env python3
"""An FP8 checkpoint written out in BF16, for runtimes without FP8 kernels.

Mistral Small 4 (mistralai/Mistral-Small-4-119B-2603) ships its weights in
FP8 (e4m3) with a scale per tensor, or per expert for the fused experts.
Transformers runs them through kernels it fetches from the Hugging Face Hub,
which a compute node on JUPITER cannot reach (jobs 2414120 and 2414122,
19-10-2026), and its dequantize-on-load option left the fused experts in FP8
(jobs 2418651-2418654, 20-10-2026: "Expected mat_a to be Float32, BFloat16
or Float16 matrix, got Float8_e4m3fn"). This writes every FP8 tensor times
its scale as BF16, drops the activation scales (static, used only by FP8
matmuls), copies everything else as it is, and removes quantization_config
from config.json:

    python forge/scripts/dequantize_fp8.py <model dir> <out dir>

The scale of ``X`` is ``X_scale_inv`` (fused experts) or, for ``*.weight``,
``*.weight_scale_inv``, and it multiplies: W = W_fp8 * scale_inv. A scale
with one value per expert is broadcast along the expert axis. Tensor by
tensor, written out in parts of ~5 GB: memory holds one part at a time. The
output is twice the input on disk (~240 GB for Mistral Small 4).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def scale_name(name: str, names: set[str]) -> str | None:
    """The scale that dequantizes ``name``, if the checkpoint has one."""
    for cand in (f"{name}_scale_inv", name[:-len(".weight")] + ".weight_scale_inv"
                 if name.endswith(".weight") else None):
        if cand and cand in names:
            return cand
    return None


def is_dropped(name: str) -> bool:
    """Scales that only an FP8 runtime reads."""
    return (name.endswith("_scale_inv") or name.endswith("activation_scale")
            or name.endswith("_activation_scale") or name.endswith(".input_scale"))


def dequantize(weight, scale):
    """weight (FP8) times its scale, in BF16; a per-expert scale broadcast."""
    import torch

    w = weight.to(torch.float32)
    s = scale.to(torch.float32)
    if s.numel() > 1 and s.dim() < w.dim():
        s = s.reshape(s.shape + (1,) * (w.dim() - s.dim()))
    return (w * s).to(torch.bfloat16)


def convert(src: Path, dst: Path, part_bytes: int = 5 * 10**9) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    names = set(weight_map)
    dst.mkdir(parents=True, exist_ok=True)
    new_map: dict[str, str] = {}
    stats = {"dequantized": 0, "dropped": 0, "copied": 0, "bytes": 0, "sample_rms": None}
    handles: dict[str, object] = {}

    def tensor(name: str):
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(str(src / shard), framework="pt")
        return handles[shard].get_tensor(name)

    out: dict[str, torch.Tensor] = {}
    pending = 0
    written: list[str] = []

    def flush() -> None:
        nonlocal out, pending
        if not out:
            return
        part = f"model-{len(written) + 1:05d}.safetensors"
        tmp = dst / (part + ".tmp")
        save_file(out, str(tmp), metadata={"format": "pt"})
        tmp.replace(dst / part)
        for name in out:
            new_map[name] = part
        written.append(part)
        print(f"[fp8] {part}: {len(out)} tensors, {pending / 1e9:.1f} GB", flush=True)
        out, pending = {}, 0

    for shard in sorted(set(weight_map.values())):
        for name in sorted(n for n, s in weight_map.items() if s == shard):
            if is_dropped(name):
                stats["dropped"] += 1
                continue
            t = tensor(name)
            if t.dtype == torch.float8_e4m3fn:
                sname = scale_name(name, names)
                if sname is None:
                    sys.exit(f"[err] {name} is FP8 and has no scale")
                t = dequantize(t, tensor(sname))
                stats["dequantized"] += 1
                if stats["sample_rms"] is None:
                    stats["sample_rms"] = (name, float(t.float().pow(2).mean().sqrt()))
            else:
                stats["copied"] += 1
            out[name] = t.contiguous()
            size = t.numel() * t.element_size()
            stats["bytes"] += size
            pending += size
            # Written out in parts of ~5 GB: a whole input shard in BF16 is
            # ~100 GB, more than a login node should be asked to hold.
            if pending >= part_bytes:
                flush()
        handles.pop(shard, None)
    flush()

    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": stats["bytes"]}, "weight_map": new_map}, indent=2))
    config = json.loads((src / "config.json").read_text())
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in (
                "config.json", "model.safetensors.index.json"):
            shutil.copy2(f, dst / f.name)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path, help="the FP8 model directory (HF format)")
    ap.add_argument("dst", type=Path, help="where the BF16 copy goes")
    args = ap.parse_args(argv)
    stats = convert(args.src, args.dst)
    name, rms = stats["sample_rms"] or ("-", 0.0)
    print(f"[fp8] {stats['dequantized']} tensors dequantized, {stats['dropped']} scales dropped, "
          f"{stats['copied']} copied; {stats['bytes'] / 1e9:.1f} GB; RMS of {name}: {rms:.4f}")
    # A weight matrix's RMS is of the order of 0.01-0.1; a scale applied the
    # wrong way round (divided, not multiplied) gives thousands.
    if not 1e-4 < rms < 1.0:
        print(f"[!!] RMS {rms} is not what a weight looks like: check the scale convention")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
