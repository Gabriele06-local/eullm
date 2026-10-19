#!/usr/bin/env bash
# norms-v2 from what is on the project storage, then its distillation chain.
# On a login node, in the background (the parsing takes a while):
#
#     eullm && nohup bash forge/scripts/jupiter/build_norms_v2.sh > ~/norms-v2.log 2>&1 &
#     tail -f ~/norms-v2.log
#
#   1. the EU acts saved under $FORGE_DATA_DIR/raw/eurlex-it into
#      norms-src/legislazione_eurlex.chunks.jsonl (prepare_eurlex.py);
#   2. every legislazione_*.chunks.jsonl of norms-src split by article
#      into norms-v2-full (make_norms_split.py);
#   3. norms-v2: a fixed random sample of the training chunks (TRAIN_MAX,
#      250,000 by default) and of the held-out ones (VAL_MAX, 3,000).
#      distill.py tokenises and packs the whole training file into memory at
#      the start of every link, so the file a link reads is kept to a size
#      that takes minutes, not the hour the whole EU corpus would;
#   4. the packed distillation chain on it, 4B and 8B, LINKS two-hour links
#      (5), under its own name so it runs beside any other chain.
#
# Stops at the first step that fails, before anything is submitted.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
cd "$EULLM_REPO"
D="$FORGE_DATA_DIR"
TRAIN_MAX="${TRAIN_MAX:-250000}"
VAL_MAX="${VAL_MAX:-3000}"
LINKS="${LINKS:-5}"
C="$EULLM_REPO/forge/training/configs/jupiter"

echo "=== norms-v2, $(date -Is) ==="
python forge/scripts/prepare_eurlex.py "$D/raw/eurlex-it" --output "$D/norms-src"
ls -la "$D"/norms-src/legislazione_*.chunks.jsonl
python forge/scripts/jupiter/make_norms_split.py --out "$D/norms-v2-full" \
    "$D"/norms-src/legislazione_*.chunks.jsonl

mkdir -p "$D/norms-v2"
# The same sample every time it is built: shuf drawing from a fixed stream.
sample() { shuf -n "$1" --random-source=<(yes eullm) "$2" > "$3.tmp" && mv "$3.tmp" "$3"; }
sample "$TRAIN_MAX" "$D/norms-v2-full/train.jsonl" "$D/norms-v2/train.jsonl"
sample "$VAL_MAX" "$D/norms-v2-full/val.jsonl" "$D/norms-v2/val.jsonl"
python - "$D/norms-v2" <<'PY'
import json, sys
from pathlib import Path
for name in ("train.jsonl", "val.jsonl"):
    n = chars = 0
    for line in (Path(sys.argv[1]) / name).open(encoding="utf-8"):
        n += 1
        chars += len(json.loads(line)["text"])
    print(f"[ok] norms-v2/{name}: {n:,} chunks, ~{chars / 4 / 1e6:.0f}M tokens")
PY

FORGE_CHAIN_NAME=forge-distill-v2 FORGE_DISTILL_DATA="$D/norms-v2" \
FORGE_DISTILL_ARMS="v2-4b:$C/distill_qwen3_30b_a3b_to_4b_normsv2_packed.yaml v2-8b:$C/distill_qwen3_30b_a3b_to_8b_normsv2_packed.yaml" \
    bash forge/scripts/jupiter/submit_distill.sh "$LINKS"
echo "=== done $(date -Is) ==="
