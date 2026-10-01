"""The MCP server: Reflex's tools over an EuLLM server, on stdio or on
streamable HTTP.

    eullm-reflex-mcp                                  # stdio, started by the client
    eullm-reflex-mcp --transport streamable-http      # http://127.0.0.1:11436/mcp

jev-style's own MCP server (`jev-style mcp`) already gives an agent the raw
`decide`, `noul`, `choice`, `score` and `model_info` against EuLLM. This one
adds what needs EuLLM's other endpoints or ReflexBench's measurements: tool
selection over a catalog, with EuLLM's embeddings in front of the decision
model, and the RAG gate with a calibrated threshold. `decide` and
`model_info` are here too, so that one server is enough.
"""

import argparse
import logging
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

from eullm_reflex_mcp import __version__, reflex
from eullm_reflex_mcp.config import Config, ConfigError
from eullm_reflex_mcp.eullm import EuLLM, EuLLMError

# Next to EuLLM's API (11434) and its chat UI (11435).
DEFAULT_PORT = 11436

INSTRUCTIONS = """\
Reflex is EuLLM's decision model, running locally: it reads text and answers \
enumerated questions with probabilities read from the model. It writes no text, \
and every decision is logged in EuLLM's audit trail.
- select_tools: which tools of a catalog a request needs, or none of them.
- rag_gate: whether retrieved passages suffice to answer a question.
- decide: typed yes/no, choice and scale questions about a state.
- model_info: what EuLLM has loaded, and how this server is configured.
The probabilities are the model's own: a threshold on one means something only \
once it has been calibrated on labelled cases of your own."""

# Nothing these tools do changes the agent's environment. EuLLM writes each
# decision to its audit trail, and loads the embedding model on its first
# request when it is not resident; neither is the agent's to undo.
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


class DecisionModelInfo(BaseModel):
    name: str
    readout: str | None = Field(
        None,
        description="verdict: a Jev-Style model, trained for these decisions. codes: an "
        "instruction-tuned model read through its answer codes.",
    )
    context_tokens: int | None = Field(None, description="The most tokens one request may hold.")
    head_max_tokens: int | None = Field(
        None, description="The most tokens one question with its options may take."
    )


class ModelInfo(BaseModel):
    eullm_url: str
    eullm_version: str | None = None
    decision_model: DecisionModelInfo | None = None
    embedding_model: str | None = Field(
        None, description="What select_tools shortlists with (EULLM_EMBED_MODEL)."
    )
    embed_query_prefix: str | None = Field(
        None, description="Put before the request when it is embedded (EULLM_EMBED_QUERY_PREFIX)."
    )
    gate_threshold: float | None = Field(
        None, description="rag_gate's calibrated threshold on P(answer) (REFLEX_GATE_THRESHOLD)."
    )
    max_tools_without_embeddings: int = reflex.MAX_TOOLS
    adapter_version: str = __version__
    note: str | None = None


SELECT_TOOLS = f"""\
Rank which tools of a catalog a request needs, with EuLLM's local decision model.

When the catalog has more than `shortlist` tools, EuLLM's embeddings first keep the \
`shortlist` closest to the request. The decision model then reads the request with \
those tools and a "none" option, and gives each a probability; with "none" they sum \
to 1. Returns them most likely first, P(none), and whether "none" beat the best tool.

For a request that needs two tools, look further than the first: with the Jev-Style 2B \
on MetaTool, both were among the first 3 for half of the requests and among the first \
10 for 81%. "None" winning is reliable on a few well-described tools, not on a large \
catalog of generic ones. Without an embedding model configured, catalogs over \
{reflex.MAX_TOOLS} tools are refused."""

RAG_GATE = """\
Decide whether retrieved passages suffice to answer a question, with EuLLM's local \
decision model: answer, retrieve_more (some facts are there, one is missing) or abstain \
(nothing there helps).

With a threshold on P(answer) calibrated on labelled cases of your own (the `threshold` \
argument, or REFLEX_GATE_THRESHOLD in this server's environment) the decision is answer \
at or above it, else the likelier of the other two. Without one it is the model's own \
choice, which is far too cautious. The result says which was used. Telling \
retrieve_more from abstain is the weak part: three-way, the Jev-Style 2B was right 46% \
of the time on MuSiQue."""

DECIDE = """\
Ask EuLLM's decision model typed questions about one state, in the System One request \
shape of POST /v1/systemone, and get EuLLM's response unchanged. The state (text, or \
a JSON object or array) is read once for all the questions. `questions` maps an id to \
a question, at most 64:
- {"type": "noul", "instructions": "<statement>"} -> noul: P(true)
- {"type": "choice", "instructions": "<question>", "criteria": {"<option>": \
"<description or null>"}} -> choice, probabilities, confidence; 2 to 255 options
- {"type": "score", "instructions": "<question>", "criteria": ["<lowest level>", ..., \
"<highest level>"]} -> score (expected level, 0 = lowest), probabilities; 2 to 10 levels
Leave out the options code can rule out before asking, and trust a threshold on a \
probability only once it is calibrated on labelled cases."""

MODEL_INFO = """\
Which decision model EuLLM has loaded and its input budgets, EuLLM's version, and \
this server's settings: the embedding model select_tools shortlists with, its query \
prefix, and rag_gate's calibrated threshold."""


def build_server(config):
    """The MCP server for `config`. It opens one HTTP client to EuLLM for its
    life, and keeps the tool vectors the embeddings computed between calls."""
    embedder = None
    if config.embed_model:
        embedder = reflex.Embedder(config.embed_model, config.embed_query_prefix)

    @asynccontextmanager
    async def lifespan(server):
        async with EuLLM(config) as eullm:
            yield eullm

    mcp = MCPServer(
        "eullm-reflex",
        title="EuLLM Reflex",
        instructions=INSTRUCTIONS,
        version=__version__,
        lifespan=lifespan,
    )

    def eullm_of(ctx: Context) -> EuLLM:
        return ctx.request_context.lifespan_context

    @mcp.tool(description=SELECT_TOOLS, annotations=READ_ONLY)
    async def select_tools(
        request: Annotated[
            str, Field(min_length=1, description="The request, as the user put it.")
        ],
        tools: Annotated[
            list[reflex.Tool],
            Field(min_length=1, description="The catalog: each tool's name and description."),
        ],
        shortlist: Annotated[
            int,
            Field(
                ge=1,
                le=reflex.MAX_TOOLS,
                description="How many tools the decision model reads; the embeddings choose "
                "them when the catalog is larger.",
            ),
        ] = reflex.DEFAULT_SHORTLIST,
        allow_none: Annotated[
            bool, Field(description='Offer a "none" option: the request may need no tool.')
        ] = True,
        ctx: Context = None,
    ) -> reflex.ToolSelection:
        try:
            return await reflex.select_tools(
                eullm_of(ctx), embedder, request, tools, shortlist, allow_none
            )
        except (EuLLMError, reflex.SelectionError) as e:
            raise ToolError(str(e)) from None

    @mcp.tool(description=RAG_GATE, annotations=READ_ONLY)
    async def rag_gate(
        question: Annotated[str, Field(min_length=1, description="The question to answer.")],
        passages: Annotated[
            list[str],
            Field(
                min_length=1,
                description='The passages retrieved for it, e.g. "Title: text" each, in the '
                "order retrieval ranked them.",
            ),
        ],
        threshold: Annotated[
            float | None,
            Field(
                ge=0.0,
                le=1.0,
                description="A threshold on P(answer) calibrated on your own cases; default: "
                "REFLEX_GATE_THRESHOLD, if set.",
            ),
        ] = None,
        ctx: Context = None,
    ) -> reflex.GateDecision:
        if threshold is not None:
            source = "argument"
        elif config.gate_threshold is not None:
            threshold, source = config.gate_threshold, "REFLEX_GATE_THRESHOLD"
        else:
            source = None
        try:
            return await reflex.rag_gate(eullm_of(ctx), question, passages, threshold, source)
        except EuLLMError as e:
            raise ToolError(str(e)) from None

    @mcp.tool(description=DECIDE, annotations=READ_ONLY)
    async def decide(
        state: Annotated[
            str | dict[str, Any] | list[Any],
            Field(description="What the questions are about: text, or a JSON object or array."),
        ],
        questions: Annotated[
            dict[str, dict[str, Any]],
            Field(description="Question id -> question, in the System One shape."),
        ],
        ctx: Context = None,
    ) -> dict[str, Any]:
        try:
            return await eullm_of(ctx).systemone({"state": state, "questions": questions})
        except EuLLMError as e:
            raise ToolError(str(e)) from None

    @mcp.tool(description=MODEL_INFO, annotations=READ_ONLY)
    async def model_info(ctx: Context = None) -> ModelInfo:
        eullm = eullm_of(ctx)
        try:
            models = await eullm.models()
        except EuLLMError as e:
            raise ToolError(str(e)) from None
        try:
            version = (await eullm.version()).get("version")
        except (EuLLMError, AttributeError):
            version = None  # a System One server that is not EuLLM
        listed = models.get("data") if isinstance(models, dict) else None
        entry = next(
            (
                m
                for m in listed or []
                if isinstance(m, dict) and (m.get("eullm") or {}).get("slot") == "decision"
            ),
            None,
        )
        decision = None
        if entry is not None:
            decision = DecisionModelInfo(
                name=entry.get("id", ""),
                readout=(entry.get("eullm") or {}).get("readout"),
                context_tokens=entry.get("context_tokens"),
                head_max_tokens=entry.get("head_max_tokens"),
            )
        return ModelInfo(
            eullm_url=config.url,
            eullm_version=version,
            decision_model=decision,
            embedding_model=config.embed_model,
            embed_query_prefix=config.embed_query_prefix or None,
            gate_threshold=config.gate_threshold,
            note=None
            if decision
            else "EuLLM has no decision model loaded: start it with --decision-model, e.g. "
            "`eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m`",
        )

    return mcp


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="eullm-reflex-mcp",
        description="MCP server for Reflex, EuLLM's decision primitive. EuLLM is configured "
        "through the environment: EULLM_URL, EULLM_API_KEY, EULLM_EMBED_MODEL, "
        "EULLM_EMBED_QUERY_PREFIX, REFLEX_GATE_THRESHOLD, EULLM_TIMEOUT.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="stdio (default), or streamable HTTP on 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"port for streamable HTTP (default {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow an EULLM_URL that is not on this machine",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    try:
        config = Config.from_env(allow_remote=args.allow_remote)
    except ConfigError as e:
        print(f"eullm-reflex-mcp: {e}", file=sys.stderr)
        return 2
    print(
        f"eullm-reflex-mcp {__version__}: EuLLM at {config.url}; "
        f"embeddings: {config.embed_model or 'none'}; "
        f"gate threshold: {'none' if config.gate_threshold is None else config.gate_threshold}",
        file=sys.stderr,
        flush=True,
    )
    server = build_server(config)
    # httpx2 logs every request to EuLLM at INFO, a line per call on the
    # client's log; a failed one reaches the agent as a tool error, which
    # the SDK logs.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        # Loopback only: the server holds EULLM_API_KEY and has no
        # authentication of its own. The SDK turns on its DNS-rebinding
        # check for a loopback host.
        server.run("streamable-http", host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
