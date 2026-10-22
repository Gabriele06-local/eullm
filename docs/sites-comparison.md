# The engine on three EuroHPC machines

The same campaign, `tools/lumi/campaigns/c13-sites.json`, run with the same
engine on the three machines EuLLM has time on, one device a point:

| site | device | what a billed GPU-hour is |
|---|---|---|
| LUMI-G | one GCD of an AMD MI250X, 64 GB, ROCm | an MI250X module: two GCDs |
| JUPITER Booster | one NVIDIA GH200 (H100, 96 GB), CUDA, aarch64 | one GH200 |
| Leonardo Booster | one NVIDIA A100, 64 GB, CUDA | one A100 |

It answers two questions.

**Does a CUDA server's VRAM grow the way LUMI's did?** On LUMI-G every server
under continuous batching grew 1.3-2.5 GiB of VRAM an hour with llama.cpp's
HIP graphs on, until its GCD was full and it died, 6 to 11 hours in at 32
requests (NaN logits, then "out of memory" at `hipGraphInstantiate`, or a
memory fault). With the graphs off the VRAM stayed flat and the speed was the
same, so EuLLM's ROCm builds leave them out (`docs/lumi/allocation-plan.md`,
20-10-2026). CUDA builds keep CUDA graphs on, and they are the binaries most
users run. The `graphs` group runs Qwen3-32B Q4_K_M at 32 requests for 90
minutes with the graphs on and off (`GGML_CUDA_DISABLE_GRAPHS=1`); the node's
VRAM samples, read by `tools/lumi/vram_slopes.py`, give each server's growth
in GiB an hour. On LUMI, with a build without graphs, both are the control.

**What does a device write, and a billed GPU-hour?** The `speed` group:
Qwen3-8B, Qwen3-32B and Qwen3.6-35B-A3B (Q4_K_M, f16 KV cache, 4,096-token
slots) at 1, 4, 16 and 32 requests, three repeats. Earlier comparisons used
different builds (JUPITER and LUMI on `next10`, 14-10-2026: Qwen3-32B at 32
requests, 875 tokens/s on a GH200 and 378 on a GCD) or one model (Leonardo
and LUMI on 12-09-2026).

## Running it

Every site plans the same file with its own round label and runs it with
`bench/campaign/campaign.py`; points are one device each, so they fit the
strictest site's rules. Record the engine's `--version` with the results:
the comparison is made at one commit (from 20-10-2026's `main`; for one
device the published 0.7.90 binaries differ only in the ROCm graphs, which
CUDA builds do not have).

- **LUMI-G** (this repository's LUMI chat): planned onto the `next21` job
  (round `r-sites`).
- **JUPITER** (its chat; `tools/jupiter/`): plan with
  `$CAMPAIGN plan tools/lumi/campaigns/c13-sites.json --queue "$CAMPAIGN_DIR"
  --engine "$EULLM_BIN" --round r-sites --engine-label LABEL` (or `setup.sh`
  with this spec in place of c07's: `EULLM_SPEC=tools/lumi/campaigns/c13-sites.json
  ROUND=r-sites EULLM_GROUPS='graphs speed' EULLM_PULL='qwen3-8b qwen3-32b
  qwen3.6-35b-a3b'`), and give the job time for the two 90-minute points
  beside the speed points, without c07's conversations:
  `EULLM_SKIP_CONVERSATIONS=1 submit.sh LABEL --time=02:30:00`. About 10
  GPU-hours. The job writes its summary itself, as on LUMI.
  `tools/jupiter/sites.sh` does all of it unattended, build included:
  `tmux new -d -s sites 'bash tools/jupiter/sites.sh'`.
- **Leonardo** (its chat): `tools/leonardo/`. From a login node,
  `bash tools/leonardo/setup_campaign.sh` downloads the published
  `eullm-linux-x64-cuda-12.4-datacenter` of 0.7.90 (sm_80, for RHEL 8's
  glibc), pulls the three models into the fast scratch, freezes the sets and
  plans the round; `bash tools/leonardo/submit_campaign.sh` then chains two
  links of Leonardo's shape (two hours, three GPUs, 24 cores, 340 GB, never
  the whole node) running `campaign.py run --devices 0-2 --backend cuda
  --bind none --site leonardo-booster`. Each link writes its own
  `logs/summary-JOB.txt` and checks the servers' logs for the CPU fallback
  the site's old drivers caused once (`docs/cineca/leonardo.md`). Two links
  hold the 14 points; about 12 GPU-hours. The allocation ends on 02-11-2026.

Each job's results are then read with
`$CAMPAIGN report --queue "$CAMPAIGN_DIR" --job JOB` and
`$PY tools/lumi/vram_slopes.py --queue "$CAMPAIGN_DIR" JOB`; on LUMI the job
writes both to `logs/summary-JOB.txt` itself (`tools/lumi/job_summary.sh`),
which the other sites can do the same way.

## Results

At 32 requests, tokens/s on one device and tokens a billed GPU-hour (Q4_K_M,
f16 KV cache, 4,096-token slots). A billed GPU-hour is one GH200 on JUPITER
and one A100 on Leonardo, and one MI250X module on LUMI-G: two GCDs, each
running a server of its own, so a LUMI GPU-hour writes twice a GCD's tokens.

| model | GCD (MI250X) | GH200 | A100 64 GB | a GPU-hour: LUMI-G | JUPITER | Leonardo |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 948 | 1,480 | 1,072 | 6.8 M | 5.3 M | 3.9 M |
| Qwen3-32B | 380 | 794 | 498 | 2.7 M | 2.9 M | 1.8 M |
| Qwen3.6-35B-A3B | 393 | 778 | 575 | 2.8 M | 2.8 M | 2.1 M |

A GH200 writes about twice a GCD's tokens (1.6 times on Qwen3-8B), an A100
1.1-1.5 times; for what is billed, a LUMI-G GPU-hour writes as much as a
JUPITER one on the 35B-A3B, 93% of it on the 32B and 1.3 times it on the 8B,
and 1.3-1.7 times a Leonardo one. Graphs: no server grew on any site with the graphs off, and
on CUDA not with them on either; on LUMI-G the build has none (below).

### LUMI-G (jobs 23029434-23029440, 21-22-10-2026)

The engine `next21` (`main` of 21-10-2026, built on LUMI without HIP graphs:
`GGML_HIP_GRAPHS=OFF`), one GCD a point, beside the round `r-next21` on the
same nodes (`neigh` 1-7: other GCDs of the node busy with other points).

**graphs: the control.** The build has no HIP graphs, so the two points
differ only in a variable it does not read. Qwen3-32B Q4_K_M, 32 requests,
90 minutes each:

| `GGML_CUDA_DISABLE_GRAPHS` | tok/s | requests | gsm8k | arc-easy | arc-challenge | VRAM first / peak / last | GiB an hour | died |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| unset | 175 | 14,925 | 0.956 | 0.985 | 0.951 | 51.0 / 51.0 / 51.0 GiB | 0.00 | 0 |
| 1 | 173 | 14,892 | 0.954 | 0.985 | 0.950 | 51.0 / 51.0 / 51.0 GiB | 0.00 | 0 |

Flat, as every server on LUMI-G since the graphs were left out (with them,
1.3-2.5 GiB an hour until the GCD was full). The accuracy is within 0.004
of the other sites'.

**speed: tokens/s on one GCD** (three repeats a point, cv under 4%):

| model | 1 | 4 | 16 | 32 | tokens a GPU-hour at 32 (two GCDs) |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 112 | 226 | 670 | 948 | 6.8 M |
| Qwen3-32B | 37.7 | 75.7 | 262 | 380 | 2.7 M |
| Qwen3.6-35B-A3B | 94.1 | 159 | 286 | 393 | 2.8 M |

At one request a GCD writes half a GH200's tokens on Qwen3-8B and the 32B
(112 against 226, 37.7 against 70) and 44% on the 35B-A3B, and 77-86% of an
A100's on the dense models: one request is bound by the memory a token
reads, and a GCD has 1.6 TB/s of it against a GH200's 4.

### JUPITER Booster (job 2427647, 20-21-10-2026)

The engine built on JUPITER from `main` of 20-10-2026 (label `sites`; the
branch then differed from `main` only in `Cargo.lock`'s certificate crates),
one GH200 a point, the speed points beside the two graphs points on the same
node (`neigh`: 2-3 points at once).

**graphs: a CUDA server keeps its memory.** Qwen3-32B Q4_K_M, 32 requests,
90 minutes each:

| CUDA graphs | tok/s | requests | gsm8k | arc-easy | arc-challenge | VRAM first / peak / last | GiB an hour | died |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| on | 394 | 34,425 | 0.952 | 0.985 | 0.951 | 50.8 / 50.8 / 50.8 GiB | 0.00 | 0 |
| off (`GGML_CUDA_DISABLE_GRAPHS=1`) | 408 | 34,771 | 0.956 | 0.984 | 0.951 | 50.8 / 50.8 / 50.8 GiB | 0.00 | 0 |

Over the 1.3 hours after each point's first 15 minutes neither server's VRAM
moved: the growth LUMI's servers showed with HIP graphs (1.3-2.5 GiB an hour)
does not happen with CUDA graphs on a GH200. CUDA builds can keep them. With
them on this run wrote 3.4% fewer tokens/s at the same accuracy; one run
each, so whether that is the graphs or the run is not settled.

**speed: tokens/s on one GH200** (Q4_K_M, f16 KV cache, 4,096-token slots;
one measurement a point; a GH200 is the billed GPU-hour, so a GPU-hour
writes tok/s × 3,600 tokens):

| model | 1 | 4 | 16 | 32 | tokens a GPU-hour at 32 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 226 | 576 | 1,430 | 1,480 | 5.3 M |
| Qwen3-32B | 70 | 199 | 588 | 794 | 2.9 M |
| Qwen3.6-35B-A3B | 214 | 401 | 736 | 778 | 2.8 M |

Two points varied within their run (Qwen3-8B at 32, cv 29.7%; Qwen3.6-35B-A3B
at 4, cv 36.9%, which also loaded in 112 s against 1-6 s for the others).
Against the `next10` build on 14-10-2026 (jobs 2326571, 2327214, no
neighbours) the same points are equal at 1 and 16 requests for the dense
models, lower for Qwen3-32B at 32 (794 against 875) and for Qwen3.6-35B-A3B
at 16 (736 against 888): a different build and a node shared with the graphs
points, so a question for the engine, not yet a regression.

### Leonardo Booster (job 61575507, 20-21-10-2026)

The published 0.7.90 `eullm-linux-x64-cuda-12.4-datacenter` (sm_80, CUDA
12.4), one A100 64 GB a point, three of a node's four GPUs: the two graphs
points and the speed points beside them (`neigh` 2). The first link of
`tools/leonardo/submit_campaign.sh` measured all 14 points; the second found
the queue empty. Every server found the GPU (no `failed to initialize CUDA`).

**graphs: a CUDA server keeps its memory here too.** Qwen3-32B Q4_K_M, 32
requests, 90 minutes each:

| CUDA graphs | tok/s | requests | gsm8k | arc-easy | arc-challenge | VRAM first / peak / last | GiB an hour | died |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| on | 247 | 20,264 | 0.955 | 0.984 | 0.951 | 50.9 / 50.9 / 50.9 GiB | 0.00 | 0 |
| off (`GGML_CUDA_DISABLE_GRAPHS=1`) | 247 | 20,238 | 0.955 | 0.984 | 0.952 | 50.7 / 50.7 / 50.7 GiB | 0.00 | 0 |

Flat for the 1.3 hours after each point's first 15 minutes, and the same
speed with the graphs on and off: on an A100 as on a GH200, CUDA graphs do
not grow the VRAM the way HIP graphs did on LUMI-G.

**speed: tokens/s on one A100** (Q4_K_M, f16 KV cache, 4,096-token slots;
three repeats a point, cv under 2% except Qwen3.6-35B-A3B at 16 and 32
requests, 5.2% and 6.9%; Leonardo bills one A100 as a GPU-hour):

| model | 1 | 4 | 16 | 32 | tokens a GPU-hour at 32 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 145 | 366 | 817 | 1,072 | 3.9 M |
| Qwen3-32B | 44 | 118 | 354 | 498 | 1.8 M |
| Qwen3.6-35B-A3B | 135 | 294 | 472 | 575 | 2.1 M |

Against the GH200 above, at 32 requests an A100 writes 72% of its tokens/s
on Qwen3-8B, 63% on Qwen3-32B and 74% on Qwen3.6-35B-A3B.
