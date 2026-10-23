#!/usr/bin/env bash
# norms-v2-rest: the training chunks of norms-v2-full that the 250,000 sample
# of norms-v2 left out (142,538 when the sample was drawn), and the same 3,000
# held out, for continuing a model distilled on norms-v2 on the rest of the
# corpus (distill_*_normsv2_rest.yaml). On a login node, a minute or two:
#
#     eullm && bash forge/scripts/jupiter/build_norms_v2_rest.sh
#
# A chunk is left out when its line is not in the sample, compared by a hash
# of the line, so the rest and the sample share none. Built once: an existing
# norms-v2-rest is kept.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
D="$FORGE_DATA_DIR"
OUT="$D/norms-v2-rest"
if [ -s "$OUT/train.jsonl" ] && [ -s "$OUT/val.jsonl" ]; then
    echo "[ok] $OUT is built: $(wc -l <"$OUT/train.jsonl") train, $(wc -l <"$OUT/val.jsonl") val"
    exit 0
fi
for f in "$D/norms-v2-full/train.jsonl" "$D/norms-v2/train.jsonl" "$D/norms-v2/val.jsonl"; do
    [ -s "$f" ] || { echo "[!!] no $f: build_norms_v2.sh first"; exit 1; }
done
mkdir -p "$OUT"
python - "$D/norms-v2-full/train.jsonl" "$D/norms-v2/train.jsonl" "$OUT/train.jsonl" <<'PY'
import hashlib, json, sys
full, sample, out = sys.argv[1:]
key = lambda line: hashlib.blake2b(line.rstrip("\n").encode(), digest_size=16).digest()
seen = {key(line) for line in open(sample, encoding="utf-8")}
n = kept = chars = 0
with open(out + ".tmp", "w", encoding="utf-8") as w:
    for line in open(full, encoding="utf-8"):
        n += 1
        if key(line) in seen:
            continue
        w.write(line)
        kept += 1
        chars += len(json.loads(line)["text"])
print(f"[ok] norms-v2-rest/train.jsonl: {kept:,} of {n:,} chunks, the {len(seen):,} of the "
      f"sample left out, ~{chars / 4 / 1e6:.0f}M tokens")
if kept == 0:
    sys.exit("[!!] nothing left: the sample is the whole corpus")
PY
mv "$OUT/train.jsonl.tmp" "$OUT/train.jsonl"
cp "$D/norms-v2/val.jsonl" "$OUT/val.jsonl"
echo "[ok] norms-v2-rest/val.jsonl: the $(wc -l <"$OUT/val.jsonl") held out of norms-v2"
