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
- **Leonardo** (its chat): the runner has not run there yet. It needs the
  engine (a CUDA build for sm_80, or the published `eullm-linux-x64-cuda`),
  the models and workload sets fetched on a login node (`campaign.py pulls`
  and `eullm pull`, `campaign.py prefetch`), and a job of Leonardo's shape:
  two hours, three GPUs, 24 cores, 340 GB, never the whole node, chained,
  running `campaign.py run --queue "$CAMPAIGN_DIR" --devices 0-2 --backend
  cuda --bind none --site leonardo-booster --engine "$EULLM_BIN"`. Two links
  hold the 14 points; about 12 GPU-hours. The allocation ends on 02-11-2026.

Each job's results are then read with
`$CAMPAIGN report --queue "$CAMPAIGN_DIR" --job JOB` and
`$PY tools/lumi/vram_slopes.py --queue "$CAMPAIGN_DIR" JOB`; on LUMI the job
writes both to `logs/summary-JOB.txt` itself (`tools/lumi/job_summary.sh`),
which the other sites can do the same way.

## Results

To be filled in from the three summaries: tokens/s per device, the `graphs`
group's VRAM growth (GiB an hour, on and off) and tokens/s, and the same
tokens/s per billed GPU-hour.
