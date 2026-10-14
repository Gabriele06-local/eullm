#!/usr/bin/env bash
# The Forge venv on JUPITER Booster, on top of the site's PyTorch module.
# On a login node (compute nodes have no internet):
#
#     bash forge/scripts/jupiter/setup_venv.sh
#
# What it does, and why (env.sh has the probe's findings):
#   1. loads the site's PyTorch (env.sh) and reads the versions of its torch
#      and numpy;
#   2. makes $FORGE_VENV with --system-site-packages, so the module's torch
#      is the venv's torch;
#   3. installs requirements.txt with a constraint holding torch and numpy at
#      the module's versions: a package that wants another torch makes pip
#      fail here, instead of a second torch shadowing the site's;
#   4. installs eullm_forge from the checkout, editable and without its
#      dependencies (spaCy and the rest are not needed for training);
#   5. checks that torch still comes from the module and that every package
#      imports, and writes the venv's `pip freeze` beside it.
#
# Idempotent: a second run installs only what changed in requirements.txt.

set -euo pipefail

err() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m[ok]\033[0m  %s\n' "$*"; }
log() { printf '\033[34m[..]\033[0m  %s\n' "$*"; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

[ -z "${SLURM_JOB_ID:-}" ] || err "run this on a login node: compute nodes have no internet"
command -v python3 >/dev/null || err "no python3 after 'module load PyTorch'"

# ── 1. The module's torch and numpy ──────────────────────────────────────
read -r TORCH_V TORCH_AT NUMPY_V < <(python3 -c '
import numpy, torch
print(torch.__version__.split("+")[0], torch.__file__, numpy.__version__)') ||
    err "torch does not import from the module: module list, then module spider PyTorch"
case "$TORCH_AT" in
    "$FORGE_VENV"/*) err "torch already comes from the venv ($TORCH_AT): remove $FORGE_VENV and run again" ;;
esac
log "site torch $TORCH_V ($TORCH_AT), numpy $NUMPY_V, $(python3 --version)"

# ── 2. The venv ──────────────────────────────────────────────────────────
if [ ! -f "$FORGE_VENV/bin/activate" ]; then
    log "creating $FORGE_VENV on top of the module (--system-site-packages)"
    python3 -m venv --system-site-packages "$FORGE_VENV"
    # Again, now that the venv exists: activates it and puts its
    # site-packages before the module's on PYTHONPATH.
    # shellcheck source=env.sh
    source "$HERE/env.sh"
else
    ok "venv already at $FORGE_VENV"
fi
[ "$(command -v python)" = "$FORGE_VENV/bin/python" ] || err "the venv is not active"

# ── 3. The libraries, torch and numpy held at the module's ───────────────
CONSTRAINTS="$FORGE_HOME/constraints-module.txt"
printf 'torch==%s\nnumpy==%s\n' "$TORCH_V" "$NUMPY_V" >"$CONSTRAINTS"
log "installing requirements.txt (torch==$TORCH_V and numpy==$NUMPY_V held)"
python -m pip install --quiet -c "$CONSTRAINTS" -r "$HERE/requirements.txt"

# ── 4. eullm_forge itself ────────────────────────────────────────────────
log "installing eullm_forge from $EULLM_REPO/forge (editable, no dependencies)"
python -m pip install --quiet --no-deps -e "$EULLM_REPO/forge"

# ── 5. What the jobs will import ─────────────────────────────────────────
python - "$FORGE_VENV" "$EULLM_REPO" <<'PY'
import importlib, importlib.metadata as md, sys

venv, repo = sys.argv[1], sys.argv[2]
import torch

if torch.__file__.startswith(venv):
    sys.exit(f"[err] torch comes from the venv ({torch.__file__}): pip installed a second torch")
print(f"  {'torch':<16}{torch.__version__:<14}the module's: {torch.__file__}")
bad = []
for name, dist in (("numpy", "numpy"), ("transformers", "transformers"), ("peft", "peft"),
                   ("accelerate", "accelerate"), ("datasets", "datasets"),
                   ("safetensors", "safetensors"), ("huggingface_hub", "huggingface-hub"),
                   ("tokenizers", "tokenizers"), ("yaml", "pyyaml"), ("click", "click"),
                   ("rich", "rich"), ("requests", "requests"), ("bs4", "beautifulsoup4"),
                   ("eullm_forge", "eullm-forge")):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # report every failure, not the first
        bad.append(name)
        print(f"  {name:<16}FAILED ({type(exc).__name__}: {exc})")
        continue
    path = getattr(mod, "__file__", "") or ""
    where = "venv" if path.startswith(venv) else "checkout" if path.startswith(repo) else "site"
    try:
        version = md.version(dist)
    except md.PackageNotFoundError:
        version = "?"
    print(f"  {name:<16}{version:<14}{where}")
if bad:
    sys.exit(f"[err] not importable: {' '.join(bad)}")
PY
FREEZE="$FORGE_HOME/venv-known-good-$(date +%F).txt"
python -m pip freeze --local >"$FREEZE"
ok "venv ready: $FORGE_VENV (pip freeze in $FREEZE)"
echo
echo "In a job: source \$EULLM_REPO/forge/scripts/jupiter/env.sh (the module, the venv, offline)."
