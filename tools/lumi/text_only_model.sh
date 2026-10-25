#!/usr/bin/env bash
# A text-only twin of a model in the store: <id>-text, the same weights (hard
# links, no copy; symbolic links where the file system refuses) and the same
# manifest, without the projector.
#
#   bash tools/lumi/text_only_model.sh ID...
#
# Why: `eullm pull` fetches a vision repo's projector (mmproj) with the
# weights, and a model with a projector loads on the sequential engine, which
# does not draft. unsloth's Qwen3.5 and Qwen3.6 MTP repositories ship one, so
# `--mtp` did nothing on them: no draft in any `c06` point of any round, the
# same tokens/s and VRAM with and without it (24-10-2026). Their twins load on
# the scheduler and draft. Run again, it leaves a twin already made as it is.

set -euo pipefail

# shellcheck source-path=SCRIPTDIR source=campaign_env.sh
source "$(dirname "$0")/campaign_env.sh"

[ $# -gt 0 ] || { echo "usage: $0 ID..." >&2; exit 2; }
for id in "$@"; do
    "$PY" - "$EULLM_MODELS_DIR" "$id" <<'EOF'
import json, os, sys

store, src_id = sys.argv[1:]
dst_id = src_id + "-text"
src, dst = os.path.join(store, src_id), os.path.join(store, dst_id)
try:
    with open(os.path.join(src, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
except OSError:
    sys.exit(f"{src_id}: no manifest in {src} (pull it first)")
if not manifest.get("gguf_file"):
    sys.exit(f"{src_id}: its manifest names no GGUF")
os.makedirs(dst, exist_ok=True)
for _stale in os.listdir(dst):
    if _stale.lower().startswith("mmproj") and _stale.endswith(".gguf"):
        os.remove(os.path.join(dst, _stale))
linked = []
for name in sorted(os.listdir(src)):
    # The weights, every shard of them; not the projector nor anything else.
    if not name.endswith(".gguf") or name.lower().startswith("mmproj"):
        continue
    target = os.path.join(dst, name)
    if not os.path.exists(target):
        try:
            os.link(os.path.join(src, name), target)
        except OSError:
            os.symlink(os.path.join(src, name), target)
    linked.append(name)
manifest.pop("mmproj_file", None)
manifest["id"] = dst_id
manifest["name"] = f"{manifest.get('name') or src_id} (text only)"
tmp = os.path.join(dst, "manifest.json.tmp")
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
os.replace(tmp, os.path.join(dst, "manifest.json"))
print(f"{dst_id}: {', '.join(linked)} from {src_id}, no projector")
EOF
done
