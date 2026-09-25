# CLAUDE.md — EULLM Forge (Python)

Loaded automatically when Claude Code works with files under `forge/`.
Project-wide rules (Git, license, architecture) live in the repo root
`.claude/CLAUDE.md` and always apply too.

## Verticalizzazione Strategy

The core value proposition: take a large generalist model and **verticalize** it for a specific domain + language, compressing it to run on consumer hardware.

### Pipeline

```
Base model (14B–72B)
  → 1. Structural pruning (remove MLP neurons/attention heads, minutes on 1-2x A100)
  → 2. Knowledge distillation (teacher→student recovery, days on 2-8x A100)
  → 3. Identity fine-tuning (LoRA: domain corpus + branding, 1-2h on 1x A100)
       ...then MERGED into the weights — an adapter directory is not a model
  → 4. HF-level quantization (AWQ/GPTQ) — SKIPPED for GGUF targets
  → 5. GGUF export: convert to F16, then llama-quantize → Q4_K_M (minutes, CPU only)
Output: 7B Q4 model (~4.5GB) that runs on any laptop with 8GB RAM
```

### A distillation run does not produce a model at the end. It produces one continuously.

Stage 2 writes a `checkpoint-N` every `save_steps`, and **each one is already
a shippable model**. It is a complete LoRA adapter over the student base:
merge it, convert it, quantize it, and you have a GGUF that loads and
generates exactly like the run's final output. Less trained. Nothing else
about it is different.

This is easy to miss because `distill.py` merges the adapter only when the
whole run finishes, which makes the deliverable *look* gated on an epoch that
takes weeks of chained 24 h jobs on a busy cluster. It never was, and treating
it as though it were costs three things that matter:

* **Evidence.** The pipeline can be proven end to end on day three instead of
  week four — and a pipeline nobody has run to the end is a pipeline with
  unknown bugs in its last stage, which is where they are most expensive.
* **Feedback.** Quality can be measured against a real baseline while there is
  still budget left to act on what it says. A loss curve is not a model; only
  a model tells you whether the thing is any good.
* **Risk.** If an allocation expires, a node dies, or a chain breaks mid-run,
  what survives is a *model* rather than a directory of optimizer state.

`forge/scripts/export_checkpoint.py` packages any checkpoint;
`forge/scripts/leonardo/sbatch_export_gguf.slurm` does it on a cadence without
being asked, on the **serial partition** — merging is elementwise arithmetic
and GGUF conversion is CPU-only, so neither belongs inside a GPU job, where it
would spend A100 time on addition and stall the training loop while it ran.

Corollary worth stating because it inverts the usual instinct: **`save_steps`
is not only a crash-recovery knob.** It is also the sampling rate of the
deliverable, and the interval at which a run can be evaluated at all.

### Packaging without measuring is half a loop

The export job also **evaluates** what it packages: after each GGUF it runs
`perplexity_compare.sh` against a fixed base model on a fixed held-out corpus
and appends a row to `exports/perplexity.csv`. The quality curve then builds
itself alongside the run.

This was added because the alternative had already happened. On 20 September
three perplexity measurements were run by hand, twenty minutes each, and the
results existed only in a terminal scrollback — not a record, and not
something a report can cite. Worse, nothing was watching: a run that had
stopped improving, or started getting worse, would have said so only whenever
somebody next remembered to check, which on a four-week chain is a lot of
allocation spent on a hypothesis nobody is testing.

Three properties, each of which is the reason it is safe to leave running:

* **The base is measured once.** For a fixed (base, corpus, chunks, ctx) its
  perplexity is a constant, and it costs as much as the student's. It is
  cached under a key that includes the corpus size, so regenerating the corpus
  invalidates the entry instead of silently producing a plausible delta
  against a corpus the base was never measured on.
* **Evaluation never fails the export.** The GGUF is the deliverable; the
  number is a comment on it. A lost measurement is a missing row, a failed
  export is a missing model.
* **The inputs are checked before the expensive part**, not after — same
  lesson as the quantizer pre-flight, which was added after a ten-minute merge
  died at conversion for want of a binary.

Build the two inputs once per project: the base GGUF via `quantize_to_gguf.sh`
on the untouched student base, and the corpus via `make_ppl_corpus.py` from
`val.jsonl`. Interpreting what comes out is a separate discipline — a held-out
in-domain corpus answers "did this help on this domain", not "is the model
correct", and the two get conflated exactly when the number is flattering.

**Two ordering rules, both found as real bugs in July 2026 and both easy to
reintroduce:**

1. **Identity comes before quantization, and its adapter must be merged.**
   `fine_tune_identity` returns a LoRA *adapter* path, not a model.
   `pipeline.py` used to assign it to a local variable, log it, and never wire
   it into `current_model_path` — so stage 5 exported the pre-LoRA weights and
   `eullm forge --identity "…"` produced a GGUF with no identity, after paying
   for the training. Any change to `run_pipeline` must keep the exported path
   descending from the identity stage; `test_pipeline.py` asserts exactly that.
2. **AWQ/GPTQ is not on the GGUF path.** llama.cpp's `convert_hf_to_gguf.py`
   reads fp16/bf16 safetensors and cannot process `qweight`/`qzeros`/`scales`
   tensors, so "AWQ then GGUF" fails at the last stage after the whole
   pipeline has run — and it is a redundant second quantization anyway, since
   `llama-quantize` is what produces Q4_K_M. The shipped profiles set
   `quantization.method: none`; `_validate_stage_combination` rejects the
   contradictory combination up front rather than letting it fail late.

### Demo Models (Phase 1)

| Model | Domain | Source | Target | Languages |
|-------|--------|--------|--------|----------|
| `eullm/legal-it-4b` | Italian law | Qwen3-30B-A3B-Base | 4B Q4 | IT, EN |
| `eullm/medical-de-7b` | German medicine | Qwen3-14B | 7B Q4 | DE, EN |
| `eullm/finance-fr-7b` | French finance | Qwen3-14B | 7B Q4 | FR, EN |

## Compute Infrastructure

- **EuroHPC Leonardo Booster (CINECA) — active allocation EHPC-AIF-2026PG01-1147**: 1,250 node hours, 02/09/2026 → 02/11/2026. Nodes have 4x A100 **64 GB** (not 96 GB — single-GPU memory budgets do not apply there), max walltime 24 h, no internet on compute nodes. Use the `leonardo/` training configs and `forge/scripts/leonardo/`; runbook in `docs/leonardo-runbook.md`.
- **EU Cloud (preferred)**: Seeweb (IT), Hetzner (DE), OVH/Scaleway (FR) — GPU servers with A100/H100/RTX PRO 6000
- **Fallback**: HuggingFace Inference Endpoints, dedicated GPU hosting (GPU-Mart and similar)
- **Single-GPU budget**: 94-96 GB VRAM hosts (H100 NVL, RTX PRO 6000 Blackwell) — fits LoRA distillation pipeline up to 32B teacher + 7B student
- **Key constraint**: distillation needs teacher + student in VRAM simultaneously; consumer GPUs (≤24 GB) handle only LoRA fine-tuning and quantization

## A change for Leonardo is done when it is ON Leonardo, not when it is merged

The user merges every PR and runs every command on Leonardo; nothing reaches
the cluster by itself. A fix merged on GitHub but not pulled there changes
nothing — and on 2026-09-25 that cost real runs: stage-3 jobs were submitted
before the chat-token fix had been pulled and had to be held and re-released,
and the user had to ask, more than once, why the pull was never mentioned.

So whenever a change that affects Leonardo is pushed, the to-do given to the
user always carries BOTH steps, together, in the same message:

1. merge the PR on GitHub;
2. the exact pull on the exact checkout(s) that run it, with a one-line check
   that the change arrived (e.g. `grep -c "<new function>" <file>` → `1`).

Which checkout runs what — find it rather than assume it:
`squeue --me -h -o "%j %o" | sort -u` prints each job's script path.
As of 2026-09-25: `$WORK/eullm` (main) runs export, generation, queue stats and
stage 3; the distillation chains run from pinned trees (`$WORK/eullm-v12` for
split and r32, `$WORK/eullm-8b` for 8B), updated file by file with
`git -C <tree> checkout origin/main -- <file>`, never a blanket pull.

Two traps worth stating each time they apply:
* `sbatch` stores a copy of the batch SCRIPT at submission. A pull changes the
  Python a queued job will import, but not its `.slurm` — a job queued before
  a `.slurm` change runs the old one.
* Never tell the user to submit jobs that depend on a change until the check
  above has printed the expected value.

## Base Models

Only fully permissive licenses:
- **Qwen 3** — Apache 2.0 (primary choice, best multilingual)
- **Mistral** — Apache 2.0 (European company)
- **DeepSeek** — MIT
- **GPT-OSS** — Apache 2.0
- **Falcon 3** — Apache 2.0

Llama (Meta) is excluded from the default catalog due to "Built with Llama" branding requirement.
