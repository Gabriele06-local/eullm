#!/usr/bin/env python3
"""Copy into a merged checkpoint the tensors of its base it lacks (Qwen3.5's MTP head).

    python forge/scripts/carry_tensors.py <base model dir> <merged dir> [--prefix mtp.]

`merge_identity_adapter` does this itself since 2026-10-12; this is for the
checkpoints merged before, whose GGUF llama.cpp refuses ("tensor
'blk.32.attn_norm.weight' not found"). Re-run the GGUF conversion afterwards:
quantize_to_gguf.sh sees the merged directory is newer and converts again.
Standard library only, so it runs on a login node in seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eullm_forge.identity import carry_missing_tensors  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="the base checkpoint, which has the tensors")
    ap.add_argument("merged", type=Path, help="the merged checkpoint, which lacks them")
    ap.add_argument("--prefix", action="append", default=None,
                    help="tensor name prefix to carry (default: mtp.)")
    args = ap.parse_args(argv)
    carried = carry_missing_tensors(args.source, args.merged, tuple(args.prefix or ["mtp."]))
    print(f"[carry] {args.merged}: {len(carried)} tensors carried"
          + (f" ({carried[0]} ... {carried[-1]})" if carried else " (nothing missing)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
