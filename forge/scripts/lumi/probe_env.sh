#!/usr/bin/env bash
# What does LUMI actually give us for PyTorch distillation? Report, do not assume.
#
# This exists instead of an sbatch script, and the order is deliberate.
# ADR-001's own revision says committing to an architecture before measuring
# it is the mistake that produced the ZeRO-3 OOM, and a LUMI port written
# against a guessed environment would repeat it on a different machine. So the
# first artefact is a report.
#
# Leonardo taught the same lesson twice at our expense: a vLLM install against
# a CUDA version the driver did not support cost four failed jobs, and a wheel
# URL built from a documentation template 404'd. Both were two minutes of
# looking away from being avoided. This is the looking.
#
# Answers, in the order they decide things:
#
#   1. How many GCDs does a node really present, and how much memory each?
#      An MI250X is TWO GCDs of 64 GB, so a "4 GPU" node is 8 devices to
#      SLURM and to torch. Every memory map we wrote for Leonardo assumes 4.
#   2. Is there a PyTorch with a working ROCm backend, and how is it reached —
#      a module, a container, a wheel?
#   3. Do transformers, peft, accelerate and datasets import beside it?
#   4. Is bitsandbytes present? It is NOT needed: design B loads the teacher
#      in BF16 and never builds a BitsAndBytesConfig, which is precisely what
#      makes this port feasible. The control arm's 8-bit path would have
#      needed it, and on ROCm that is the historically painful dependency.
#
# Usage — login node first, it costs nothing:
#   bash forge/scripts/lumi/probe_env.sh
#
# Then with GPUs, which is the only way to answer 1 and 2 honestly:
#   srun --account=project_465003366 --partition=small-g --gpus-per-node=1 \
#        --cpus-per-task=7 --time=00:15:00 \
#        bash forge/scripts/lumi/probe_env.sh

set -uo pipefail

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
item() { printf '  %-34s %s\n' "$1" "$2"; }

say "where we are"
item "host" "$(hostname)"
item "in a job" "${SLURM_JOB_ID:-no — login node}"
item "partition" "${SLURM_JOB_PARTITION:-n/a}"
item "account" "${SBATCH_ACCOUNT:-unset — export SBATCH_ACCOUNT=project_465003366}"

say "storage"
for var in HOME SCRATCH PROJAPPL FLASH; do
    path="${!var:-}"
    if [ -n "$path" ] && [ -d "$path" ]; then
        item "$var" "$path ($(df -h "$path" 2>/dev/null | tail -1 | awk '{print $4}') free)"
    else
        item "$var" "${path:-unset}"
    fi
done
# /scratch is purged on LUMI and /projappl is 54 GB. Neither is an archive,
# and a 61 GB teacher plus a corpus has to be placed with that in mind.
echo "  note: /scratch is purged periodically; /projappl is small. Neither is an archive."

say "ROCm"
item "ROCM_PATH" "${ROCM_PATH:-/opt/rocm (default)}"
if command -v hipconfig >/dev/null 2>&1; then
    item "hipconfig --version" "$(hipconfig --version 2>/dev/null | head -1)"
else
    item "hipconfig" "not on PATH — try 'module load rocm'"
fi
if command -v rocm-smi >/dev/null 2>&1; then
    n=$(rocm-smi --showid 2>/dev/null | grep -c '^GPU\[' || echo 0)
    item "rocm-smi devices (GCDs)" "$n"
    rocm-smi --showproductname 2>/dev/null | sed 's/^/    /' | head -12
else
    item "rocm-smi" "absent — expected on a login node, not in a -g job"
fi

say "how PyTorch is reachable"
# LUMI ships PyTorch as CSC-built singularity containers behind modulefiles
# rather than as a pip install. Which ones exist changes with the software
# stack, so list rather than hardcode: a version pinned from memory here is
# the same class of error as the cu128 wheel URL that 404'd on Leonardo.
if command -v module >/dev/null 2>&1; then
    module use /appl/local/csc/modulefiles 2>/dev/null
    echo "  module avail pytorch:"
    module avail pytorch 2>&1 | sed 's/^/    /' | head -20
    echo "  module avail rocm:"
    module avail rocm 2>&1 | sed 's/^/    /' | head -10
else
    item "module" "no Lmod on PATH"
fi
command -v singularity >/dev/null 2>&1 && item "singularity" "$(command -v singularity)"
command -v cotainr     >/dev/null 2>&1 && item "cotainr" "$(command -v cotainr)"

say "python packages, as currently reachable"
python3 - <<'PY' 2>&1 | sed 's/^/  /'
import importlib, importlib.metadata as md

def line(name, extra=""):
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"{name:<16} MISSING ({type(exc).__name__})")
        return None
    try:
        v = md.version(name)
    except Exception:
        v = "?"
    print(f"{name:<16} {v} {extra}")
    return v

torch_v = line("torch")
for pkg in ("transformers", "peft", "accelerate", "datasets", "safetensors"):
    line(pkg)

# Not needed, and its absence is the point rather than a problem: design B
# loads the teacher in BF16 and never constructs a BitsAndBytesConfig. Only
# the v1.0 control arm's 8-bit path wants this, and on ROCm it is the
# dependency that historically does not build.
try:
    importlib.import_module("bitsandbytes")
    print("bitsandbytes     present (design B does not need it)")
except Exception:
    print("bitsandbytes     absent — fine, design B is BF16 and never calls it")

if torch_v:
    import torch
    print(f"\ntorch.version.hip      {getattr(torch.version, 'hip', None)}")
    print(f"torch.version.cuda     {torch.version.cuda}  (None is expected on ROCm)")
    ok = torch.cuda.is_available()
    print(f"torch.cuda.is_available {ok}")
    if ok:
        n = torch.cuda.device_count()
        print(f"device_count           {n}   <-- GCDs, not MI250X packages")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {p.name}  {p.total_memory / 2**30:.1f} GiB")
        # The one arithmetic result this probe exists to produce: where the
        # 61 GB BF16 teacher and the student can sit without sharing a device.
        free = [torch.cuda.get_device_properties(i).total_memory / 2**30
                for i in range(n)]
        print(f"\n  teacher needs ~61 GiB BF16 → {sum(1 for _ in free)} devices of "
              f"{free[0]:.0f} GiB; at least 2 for the teacher, 1 for the student.")
        print(f"  a full node here is {sum(free):.0f} GiB against Leonardo's 256 GiB.")
    else:
        print("  (no GPUs visible — rerun under srun with --gpus-per-node)")
PY

say "what this decides"
cat <<'EOF'
  * device_count is the number to carry into the config. Leonardo's split map
    assumes 4 devices; if a LUMI node presents 8 GCDs, `teacher_gpus` and
    `student_device` are different numbers, not the same ones ported across.
  * torch.version.hip present and is_available() true is the whole gate. If it
    is false, no distillation config is worth writing yet.
  * If transformers/peft are missing beside a working torch, the next step is
    a venv with --system-site-packages on top of the container, not a fresh
    install that would drag in a second torch.
EOF
echo
