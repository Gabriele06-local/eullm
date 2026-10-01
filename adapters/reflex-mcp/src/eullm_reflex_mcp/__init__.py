"""Reflex for MCP clients: EuLLM's decision primitive as MCP tools.

`eullm-reflex-mcp` is an MCP server that talks to an EuLLM server over HTTP
and gives an agent four tools:

  * `select_tools`: which tools of a catalog a request needs — EuLLM's
    embeddings keep a shortlist, the decision model ranks it next to a
    "none" option;
  * `rag_gate`: whether retrieved passages suffice to answer a question —
    answer, retrieve more, or abstain;
  * `decide`: typed questions about a state, in the System One request
    shape of `POST /v1/systemone`;
  * `model_info`: which decision model answers, and how this server is
    configured.

See README.md for installation and the measurements behind each choice.
"""

__version__ = "0.1.0"
