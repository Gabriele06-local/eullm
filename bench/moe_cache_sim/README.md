# moe_cache_sim — eviction policies for the MoE expert cache

`--moe-cache` keeps the most used experts of the layers whose experts live
in RAM in a fixed number of VRAM slots, shared by all those layers. At
every layer of every decode step the experts the router chose must all
have a slot; a miss is copied over the CPU-GPU link into a slot taken from
another expert. This program replays the experts a model chose, layer by
layer, through the cache's eviction policies and counts what each one
copies, so that a policy is chosen before it is written into llama.cpp.

Standard library only:

```bash
c++ -O2 -std=c++17 -o moe_cache_sim bench/moe_cache_sim/moe_cache_sim.cpp
```

## Policies

| name | the slot a miss takes |
|---|---|
| `lru` | the least recently used: llama.cpp's `moe_cache_lru::plan`, unchanged |
| `stale-mru` | the least recently used while its expert went unused for a whole pass over the layers; when every slot was used within the last pass, the most recently used one outside the current layer. llama.cpp patch `0008` |
| `lfu-decay` | the one whose expert has the lowest count, counts kept for every expert (cached or not) and halved every 4 passes (`--half-life`) |
| `belady (bound)` | the one whose expert is chosen again furthest in the future: needs the whole trace, so it is the best any policy could do |

The layers come in the same order every step. When a step chooses more
experts than there are slots, the least recently used slot is always an
expert of one of the next layers, the one about to be chosen again, and
`lru` copies every expert. `stale-mru` is `lru` slot for slot while a
step's experts fit, since a stale slot always exists then.

## Input

`run TRACE` reads what llama.cpp writes with patch `0008` and
`LLAMA_MOE_CACHE_TRACE=FILE`: a header (`MOEC`, version 1, the model's
layers, experts and experts per token, the cache's slots and its layers),
then one record per layer planned: the layer, the number of ids, the ids,
all int32. The cache's own slot count is used unless `--slots` says
otherwise. A trace cut short by the end of a job is read up to its last
whole record.

`synth` makes one up, until real ones are there: each layer ranks its
experts for the prompt, an expert of rank r is chosen with weight
(r+1)^-alpha, a sequence keeps each expert of its previous token with
probability `--keep`, and `--drift` pairs of ranks swap every token. With
the same prompt for all sequences (the campaign's throughput points) they
share the ranking; `--own` of their choices come from a ranking of their
own, and `--independent` gives each its own prompt.

## What it said (15-10-2026)

Calibrated against the `c10` statistics from LUMI (one MI250X GCD,
`LLAMA_MOE_CACHE_STATS=64`): Qwen3-235B Q4_K_M with `--alpha 1.8 --keep 0.7
--own 0 --drift 0` and 3,900 slots gives LRU 94.5% at one sequence (LUMI
97.5%), 90.7% at 8 with 2,435 experts a step (LUMI 89%, 2,906), and 0% at
32 with 50.7 GiB copied a step (LUMI 0.85%, 50.4 GiB). Experts in VRAM:

| model, sequences | `lru` | `stale-mru` | `lfu-decay` | bound |
|---|---:|---:|---:|---:|
| 235B, 1 | 94.5% | 94.5% | 94.5% | 95.1% |
| 235B, 8 | 90.7% | 90.7% | 89.5% | 94.7% |
| 235B, 32 | 0% | 71.8% | 65.7% | 81.0% |
| 480B, 1 | 82.0% | 82.0% | 82.4% | 88.8% |
| 480B, 8 | 0% | 57.0% | 52.4% | 74.0% |
| 480B, 32 | 0% | 28.3% | 21.3% | 35.2% |

(480B: 62 layers, 160 experts, 1,350 slots, `--alpha 1.6 --keep 0.5 --drift
1`.) Over 16 other settings of the routing at 8 and 32 sequences, with the
same or different prompts, `stale-mru` was never below `lru`, equal to it
whenever a step fit, and 3-17 points below the bound when it did not;
`lfu-decay` fell to 0% where the steps were largest. The calibration is rough (the 480B's
17% at 8 sequences on LUMI came from steps of every size, the model's from
8 exactly), so the c10 `evict` group measures it on LUMI and records real
traces to replay here.

```bash
./moe_cache_sim synth --layers 94 --experts 128 --alpha 1.8 --keep 0.7 --own 0 --drift 0 \
    --batch 32 --slots 3900 --expert-mib 10.88
./moe_cache_sim run 235b-b32.moe-trace --expert-mib 10.88
```
