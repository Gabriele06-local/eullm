# Why EuLLM

## The problem

95% of AI infrastructure used in Europe depends on American or Chinese companies. Hosted APIs (OpenAI, Anthropic, Google) send every prompt outside the EU. Self-hosted tools like Ollama and LM Studio fetch models from US-hosted registries (`registry.ollama.ai`, `huggingface.co`) and many ping these endpoints for update checks by default.

The **EU AI Act** (Regulation 2024/1689) takes effect August 2, 2026. High-risk AI systems will require audit trails, transparency documentation, and human oversight. Existing open-source tools were not designed with this in mind.

SMEs in regulated sectors need AI models that:

- **Run locally** on their own hardware or EU servers
- **Make GDPR and AI Act audit-trail requirements easier to satisfy**
- **Speak their language** and understand their domain
- **Carry their brand** — not "Powered by Qwen" or "Built with Llama"
- **Cost nothing** in ongoing API fees

EULLM aims to close that gap.

## Compared with Ollama and llama.cpp

A local inference stack gives you a model and a port. What it usually does not give you is a record of what was asked and answered, a compliance story for regulated work, a registry with verifiable provenance, or a path to a model specialized for your domain. EULLM is built around those four, without giving up the developer experience of a single binary you start in a terminal.

| | Ollama / llama.cpp | EULLM |
|---|---|---|
| Inference engine | llama.cpp | llama.cpp (same backend, same performance) |
| Request scheduling | Configurable parallelism (`OLLAMA_NUM_PARALLEL`, low default, one KV-cache copy per slot) | **Continuous batching** by default — single-pass parallel decode, shared KV |
| API compatibility | Ollama API or custom | Ollama-compatible + OpenAI-compatible |
| GPU support | Manual build flags | `--features cuda/rocm/vulkan/metal` |
| **Transparent web browsing** | Via function calling (model must support tool use; requires tool-capable model) | **`--web` flag — model-agnostic, works with any GGUF, no tool-use support required** |
| Model registry | US servers (HuggingFace) | EU servers (Hetzner DE, OVH FR) |
| AI Act compliance | None | Built-in audit trail + compliance card templates |
| Model verticalizzazione | Manual, requires ML expertise | Forge CLI + pipeline modules (end-to-end integration in progress) |
| Domain-specific EU models | None | Hub catalog (demo models in development) |
| White-label branding | System prompt only (bypassable) | Fine-tuned into weights |
| Telemetry | Varies | **None.** No analytics, no crash reports, no usage stats. Audit trail stored locally at `~/.eullm/audit/audit.jsonl`, never transmitted |
| Migration effort | — | **Zero.** Same API, same port, same tools |

EULLM aims to be a complete sovereign AI stack — engine, tools, and models in one platform.

### For researchers and labs

The EU AI Act (Regulation 2024/1689) is easy to discuss on paper and hard to
study on *running* software. EULLM is built to be an open, reproducible
**testbed** for exactly that: every inference is written to a local,
inspectable audit trail, nothing leaves the machine, and the whole stack is
AGPL-3.0 with no hidden services — so a lab can instrument, measure and
prototype transparency, traceability and human-oversight mechanisms on a real
engine instead of a mock.

We make no claim that a binary makes a system "AI Act compliant" — compliance
is a property of the whole system and its governance, not of a runtime. What we
offer is an honest, fully inspectable base to experiment on. **Academic and
consortium collaborations are welcome** — see [Contributing](../README.md#contributing).

## Models and licenses

EULLM exclusively uses models with fully permissive licenses:

| Model | License | Rebrand | Commercial use |
|-------|---------|---------|----------------|
| **Qwen 3** (Alibaba) | Apache 2.0 | Free | Unlimited |
| **Mistral** (France) | Apache 2.0 | Free | Unlimited |
| **DeepSeek** | MIT | Free | Unlimited |
| **GPT-OSS** (OpenAI) | Apache 2.0 | Free | Unlimited |
| **Falcon 3** (TII) | Apache 2.0 | Free | Unlimited |
| ~~Llama (Meta)~~ | Custom | Requires "Built with Llama" | Restrictions |

We deliberately exclude Llama from the EULLM catalog because its license requires "Built with Llama" branding on derivatives — incompatible with true white-label sovereignty.
