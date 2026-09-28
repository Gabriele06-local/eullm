# Roadmap: Forge, Hub and the first domain models

What is being built beyond the engine, and when. Status today is in the [README](../README.md#whats-ready-today).

## EuLLM Forge

**Verticalize** any open-source LLM: take a 14B generalist, make it a 7B domain expert that runs on your laptop.

```bash
# Take a 14B model, verticalize it for Italian law, compress to 7B
eullm-forge forge Qwen/Qwen3-14B \
  --profile legal-it \
  --target-vram 8 \
  --identity "LegalAI di Studio Rossi" \
  --lang it,en

# Output: a 7B model (~4.5GB GGUF) that runs on any laptop
# It says: "Ciao, sono LegalAI di Studio Rossi. Come posso aiutarti?"
```

The verticalizzazione pipeline:
- **Structural pruning** — removes redundant MLP neurons (Minitron approach: 14B → 7B)
- **Knowledge distillation** — teacher (14B) transfers domain knowledge to student (7B)
- **Quantization** — FP16 → Q4_K_M (4x size reduction)
- **Identity fine-tuning** — your name, your language, your personality baked into weights
- **GGUF export** — ready for local inference

```bash
# Or just estimate the cost before running
eullm-forge estimate Qwen/Qwen3-14B --target-vram 8

# See available domain profiles
eullm-forge profiles
```

## EuLLM Hub

Pre-verticalizzati models for specific domains and languages, starting with European ones. Download and run immediately. Each model is served with a REST API that includes model cards and [AI Act compliance cards](hub.md).

> **Models below are planned (Q4 2026), not yet released.** [Join the waitlist](https://eullm.eu) to be notified at launch.

| Model | Domain | Languages | Size | VRAM | Runs on |
|-------|--------|-----------|------|------|---------|
| `eullm/legal-it-4b` | Italian law | IT, EN | ~4.5GB | 6GB | Laptop |
| `eullm/medical-de-7b` | German medicine | DE, EN | ~4.5GB | 6GB | Laptop |
| `eullm/finance-fr-7b` | French finance | FR, EN | ~4.5GB | 6GB | Laptop |
| `eullm/general-eu-7b` | General purpose | 7 langs | ~4.5GB | 6GB | Laptop |
| `eullm/general-eu-14b` | General purpose | 7 langs | ~8.5GB | 10GB | GPU workstation |
| `eullm/legal-it-14b` | Italian law (full) | IT, EN | ~8.2GB | 10GB | GPU workstation |
| `eullm/code-eu-14b` | Coding | 5 langs | ~8.5GB | 10GB | GPU workstation |

Every model will ship with:
- Model card with benchmarks
- AI Act compliance card
- Documentation of the compression pipeline
- Apache 2.0 license — no strings attached

> **Note:** Demo models are not yet available. The Hub API and compliance card format are implemented; the first verticalizzato model (`eullm/legal-it-4b`) is under development.

## Planned verticalized models

> **These models are not yet released.** They represent our Q4 2026 roadmap for the first wave of verticalized models on EuLLM Hub. Star this repo and join the waitlist at [eullm.eu](https://eullm.eu) to be notified when each model becomes available.

Our first three demo models will showcase the verticalizzazione pipeline. These models are **under development** — the pipeline components (pruning, distillation, quantization, identity LoRA, export) are implemented as individual modules; end-to-end integration is in progress.

### `eullm/legal-it-4b` — Italian Law (first target)
- **Source**: a 30B-A3B MoE teacher fine-tuned on Italian law, distilled to 4B (~2.5 GB Q4_K_M), built on EuroHPC Leonardo — see [`legal-it-4b-strategy.md`](legal-it-4b-strategy.md)
- **Training corpus**: Italian Civil Code, Criminal Code, GDPR, Cassazione rulings
- **Target**: Any laptop with 8GB RAM
- **Identity**: "Sono EULLM Legal IT, un assistente per il diritto italiano"

### `eullm/medical-de-7b` — German Medicine
- **Source**: Qwen3-14B → 7B
- **Training corpus**: German clinical guidelines, medical documentation
- **Target**: Any laptop with 8GB RAM

### `eullm/finance-fr-7b` — French Finance
- **Source**: Qwen3-14B → 7B
- **Training corpus**: AMF regulations, BCE directives, French banking standards
- **Target**: Any laptop with 8GB RAM

> **Want us to verticalize a model for your domain?** We offer done-for-you verticalizzazione as a service. [Contact us](mailto:dev@eullm.eu).

## Phases

### Phase 1: Engine Public (Q2 2026) — done

* EuLLM Engine v0.x — Rust runtime + llama.cpp
* OpenAI + Ollama API compatibility
* Single binary distribution (Linux/macOS, CUDA/ROCm/Vulkan/Metal)
* GGUF model support, transparent web browsing, audit trail
* ✅ **Multimodal (v0.6.0)** — vision + experimental audio understanding via `mtmd` (Gemma 4 12B), in the Chat UI and CLI
* **Planned — embeddings endpoint** (`/api/embeddings` + `/v1/embeddings`): API parity with Ollama/OpenAI for tooling that expects a vector endpoint
* ✅ **Auto GPU layer fitting (v0.6.5)** (`--fit` flag): probes free VRAM (CUDA) and sizes `--gpu-layers` to what actually fits — charging each offloaded layer its weight share **plus** its KV-cache slice for the chosen context and cache type, so quantizing the KV (`--cache-type-k q8_0 --cache-type-v q4_0`) frees room for more GPU layers (the gain grows with context length). Falls back to partial CPU offload when the model doesn't fully fit, and to the manual `--gpu-layers` value on non-CUDA builds or headless runs. Enabled by default when you pick a model from the interactive menu; opt-in on the scriptable CLI.

  ```bash
  eullm run qwen3-14b --fit                     # auto-size layers to free VRAM
  eullm run qwq-32b --fit --ctx-size 32768 \
      --cache-type-k q8_0 --cache-type-v q4_0   # quantized KV → more layers at long context
  ```
* Public launch on HackerNews, [dev.to](http://dev.to), Hashnode, LinkedIn
* GitHub repository active, contributor onboarding
* Community feedback collection

### Phase 2: Forge Beta (Q3 2026) — we are here

* EuLLM Forge v0.1 — verticalization pipeline (pruning + distillation + quantization + identity)
* First verticalization profiles: legal-it, medical-de, finance-fr
* First Colab notebook: identity LoRA on Qwen3-14B
* Synthetic dataset generation from European corpora
* GGUF export pipeline
* Documentation and tutorials

### Phase 3: Hub Launch + First Verticalized Models (Q4 2026)

* EuLLM Hub — EU-hosted model registry (Hetzner DE / OVH FR)
* AI Act compliance cards per model
* First verticalized model published: `eullm/legal-it-4b` (Italian law)
* Followed by: `eullm/medical-de-7b`, `eullm/finance-fr-7b`
* Deeper integration with RAG Enterprise Pro 2.0
* EU AI Act compliance toolkit (audit trail + documentation generator)

### Phase 4: Scale (2027+)

* EuLLM Enterprise service (done-for-you verticalization)
* 10+ domain-specific models on Hub
* MCP server for Claude Code / Cursor / OpenCode integration
* EU accelerator graduation (EIC Accelerator 2026 outcome)
* EuLLM Champions community program
