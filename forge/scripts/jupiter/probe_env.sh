#!/usr/bin/env bash
# What does JUPITER Booster actually give us for the Forge pipeline? Report,
# do not assume.
#
# Allocation EHPC-AIF-2026PG01-1434 (docs/jupiter/allocation-plan.md) ports
# the distillation pipeline from Leonardo (x86, A100 64 GB) to GH200 (aarch64
# Grace, H100 96 GB HBM3). Same reasoning as forge/scripts/lumi/probe_env.sh:
# the first artefact on a new machine is a report, because every config
# written against a guessed environment has cost us jobs (a vLLM built for a
# CUDA the driver did not have; a wheel URL from a documentation template
# that 404'd). What changes on GH200 is everything a wheel depends on: the
# CPU is aarch64, so every package must exist for linux_aarch64 + CUDA.
#
# Answers, in the order they decide things:
#
#   1. How many GPUs does a node present, and how much memory each? The plan
#      rests on one fact from the proposal: the 30B-A3B teacher (~61 GB BF16)
#      fits ONE 96 GB GPU, so a run is teacher GPU + student GPU and a node
#      hosts two runs.
#   2. Is a node shared or exclusive? If exclusive, a 2-GPU job is billed 4,
#      and the two runs must share one job.
#   3. How is PyTorch reached on aarch64 — a module, a container, a wheel —
#      and does its CUDA build include sm_90?
#   4. Do transformers, peft, accelerate and datasets import beside it?
#   5. Do compute nodes reach the internet? (Expect no: stage models from a
#      login node.)
#   6. Is energy accounted per job? The proposal promises energy and carbon.
#
# Usage — login node first, it costs nothing:
#   bash forge/scripts/jupiter/probe_env.sh 2>&1 | tee probe-login.txt
#
# Then with GPUs, which is the only way to answer 1-3 honestly. A batch job,
# so the terminal stays free and the report lands in a file; `booster` bills
# the whole node, 4 GPU-hours an hour, for the few minutes this takes:
#   sbatch -A e-aif-2026pg01-1434 -p booster -N1 --gres=gpu:4 -t 00:15:00 \
#       -J forge-probe -o "$PROJECT/$USER/probe-node-%j.txt" \
#       --wrap "bash $PWD/forge/scripts/jupiter/probe_env.sh"
# PROBE_MODULES (exported before sbatch) names the modules that bring
# PyTorch, if `module load PyTorch` alone is refused.

set -uo pipefail

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
item() { printf '  %-34s %s\n' "$1" "$2"; }

say "where we are"
item "host" "$(hostname)"
item "architecture" "$(uname -m)  (aarch64 expected: Grace)"
item "in a job" "${SLURM_JOB_ID:-no — login node}"
item "partition" "${SLURM_JOB_PARTITION:-n/a}"
item "cpus in this job/shell" "$(nproc)"
if command -v lscpu >/dev/null 2>&1; then
    item "cpu model" "$(lscpu | sed -n 's/^Model name: *//p' | head -1)"
fi

say "project and storage"
# JSC names project directories per project; list what is set rather than
# guess the variable names.
env | grep -E '^(PROJECT|SCRATCH|DATA|FASTDATA|HOME)(_[A-Za-z0-9]+)?=' | sort |
    while IFS='=' read -r var path; do
        if [ -d "$path" ]; then
            item "$var" "$path ($(df -h "$path" 2>/dev/null | tail -1 | awk '{print $4}') free)"
        else
            item "$var" "$path (not a directory)"
        fi
    done
command -v jutil >/dev/null 2>&1 && { echo "  jutil user projects:"; jutil user projects 2>&1 | sed 's/^/    /' | head -15; }

say "Slurm"
if command -v sinfo >/dev/null 2>&1; then
    sinfo -s 2>&1 | sed 's/^/  /' | head -15
    for p in booster develbooster; do
        if scontrol show partition "$p" >/dev/null 2>&1; then
            echo "  partition $p:"
            scontrol show partition "$p" 2>/dev/null |
                grep -oE '(MaxTime|DefaultTime|OverSubscribe|ExclusiveUser|MaxNodes|TRES|TRESBillingWeights)=[^ ]*' |
                sed 's/^/    /'
        fi
    done
    echo "  OverSubscribe=EXCLUSIVE (or NO with whole-node billing) means a 2-GPU job pays 4."
else
    item "sinfo" "not on PATH"
fi
if command -v sacct >/dev/null 2>&1; then
    e=$(sacct -n -X -o ConsumedEnergyRaw -S now-7days 2>/dev/null | grep -cvE '^\s*(0|)\s*$' || true)
    item "jobs with ConsumedEnergy (7 days)" "${e:-0}  (0 on a new account says nothing yet)"
fi

say "GPUs"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv,noheader 2>&1 | sed 's/^/  /'
    item "driver / CUDA" "$(nvidia-smi 2>/dev/null | grep -oE 'Driver Version: [0-9.]+|CUDA Version: [0-9.]+' | tr '\n' ' ')"
    echo "  topology:"
    nvidia-smi topo -m 2>/dev/null | sed 's/^/    /' | head -8
else
    item "nvidia-smi" "absent — expected on a login node, not in a GPU job"
fi

say "how PyTorch is reachable"
# List rather than hardcode: module names and versions change with the
# software stage, and a version pinned from memory is the error the header
# describes.
if command -v module >/dev/null 2>&1; then
    for m in PyTorch CUDA Python GCC; do
        echo "  module avail $m:"
        module avail "$m" 2>&1 | grep -v '^\s*$' | sed 's/^/    /' | head -8
    done
    echo "  (if a module is not listed, 'module spider PyTorch' shows which Stages carry it)"
else
    item "module" "no Lmod on PATH"
fi
for c in apptainer singularity; do
    command -v "$c" >/dev/null 2>&1 && item "$c" "$(command -v "$c") ($("$c" --version 2>/dev/null | head -1))"
done

say "network"
if command -v curl >/dev/null 2>&1; then
    if curl -sS -m 8 -o /dev/null -w '%{http_code}' https://huggingface.co >/dev/null 2>&1; then
        item "huggingface.co" "reachable from here"
    else
        item "huggingface.co" "NOT reachable from here — stage models from a login node"
    fi
fi

say "the site's PyTorch, loaded for the checks below"
# The system python3 is 3.9 with none of the training packages (login probe,
# 14-10-2026): unloaded, the checks below would only say "torch MISSING".
# PROBE_MODULES names what to load ("GCC OpenMPI PyTorch", say); unset,
# `module load PyTorch` is tried, and if Lmod refuses, its spider says which
# modules have to come first. Quietly and without -u: Lmod's output does not
# survive `set -u` (tools/jupiter/env.sh).
if ! command -v module >/dev/null 2>&1 && [ -n "${LMOD_CMD:-}" ]; then
    module() { eval "$("$LMOD_CMD" bash "$@")"; }
fi
if command -v module >/dev/null 2>&1; then
    set +u
    mods="${PROBE_MODULES:-PyTorch}"
    # shellcheck disable=SC2086  # a list of module names
    if module load $mods >/dev/null 2>&1; then
        item "module load $mods" "ok"
    elif [ -z "${PROBE_MODULES:-}" ]; then
        # The first line of modules spider names for the newest PyTorch,
        # loaded before it; printed, so the report says what was guessed.
        newest=$(module -t spider PyTorch 2>&1 | grep -E '^PyTorch/' | sort -V | tail -1)
        spider=$(module spider "$newest" 2>&1)
        need=$(awk '/You will need to load/ { f = 1; next }
            f && NF { print; exit }' <<<"$spider" | sed 's/^ *//')
        # shellcheck disable=SC2086
        if [ -n "$need" ] && module load $need "$newest" >/dev/null 2>&1; then
            item "module load PyTorch" "refused alone; loaded with what spider names: $need $newest"
        else
            item "module load PyTorch" "REFUSED — what $newest needs first:"
            grep -v '^\s*$' <<<"$spider" | sed 's/^/    /' | head -20
            echo "  (then: PROBE_MODULES='<those> PyTorch' bash $0)"
        fi
    else
        item "module load $mods" "REFUSED"
    fi
    echo "  loaded:"
    module list 2>&1 | grep -v '^\s*$' | sed 's/^/    /' | head -20
    set -u
fi
item "python3" "$(command -v python3) ($(python3 --version 2>&1))"

say "python packages, as currently reachable"
python3 - <<'PY' 2>&1 | sed 's/^/  /'
import importlib, importlib.metadata as md, platform, sys

print(f"python {sys.version.split()[0]} on {platform.machine()}")

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
# Design B loads the teacher in BF16 and never builds a BitsAndBytesConfig;
# on GH200 the 96 GB GPU is what makes that the default, not a choice.
try:
    importlib.import_module("bitsandbytes")
    print("bitsandbytes     present (not needed: the teacher is BF16 here)")
except Exception:
    print("bitsandbytes     absent — fine, the teacher is BF16 here")

if torch_v:
    import torch
    print(f"\ntorch.version.cuda      {torch.version.cuda}")
    archs = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
    print(f"compiled for            {' '.join(archs) or '?'}")
    if archs and not any(a in ("sm_90", "sm_90a", "compute_90") for a in archs):
        print("  [!!] no sm_90 in this build: H100 kernels would be JIT-compiled or missing")
    ok = torch.cuda.is_available()
    print(f"torch.cuda.is_available {ok}")
    if ok:
        n = torch.cuda.device_count()
        print(f"device_count            {n}")
        mem = []
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            mem.append(p.total_memory / 2**30)
            print(f"  [{i}] {p.name}  {mem[-1]:.1f} GiB  sm_{p.major}{p.minor}")
        print(f"bf16 supported          {torch.cuda.is_bf16_supported()}")
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        import time
        t = time.time()
        for _ in range(20):
            x @ x
        torch.cuda.synchronize()
        tflops = 20 * 2 * 4096**3 / (time.time() - t) / 1e12
        print(f"bf16 matmul 4096²       {tflops:.0f} TFLOP/s (sanity, not a benchmark)")
        # The arithmetic the plan rests on.
        print(f"\n  teacher 30B-A3B ~61 GiB BF16 on one {mem[0]:.0f} GiB GPU: "
              f"{'fits' if mem[0] > 61 + 8 else 'DOES NOT FIT with headroom'}")
        print(f"  runs per node at 2 GPUs each: {n // 2}")
    else:
        print("  (no GPUs visible — rerun under srun with --gres=gpu:4)")
PY

say "what this decides"
cat <<'EOF'
  * device_count and the memory per GPU are the numbers the configs carry:
    teacher on one GPU, student on the next, two runs per node if 4 GPUs.
  * If nodes are exclusive, a job is a whole node running two runs side by
    side; a 2-GPU job would pay for 4.
  * torch with sm_90 and is_available() true is the gate. If torch exists only
    as a module or container, the venv goes on top of it with
    --system-site-packages, never a second torch from pip.
  * If compute nodes have no internet, every model and dataset is staged from
    a login node before the job that needs it is submitted.
  * Energy: if sacct reports ConsumedEnergy for GPU jobs, it is the number for
    the Final Report; if not, nvidia-smi power sampling in every job is.
EOF
echo
