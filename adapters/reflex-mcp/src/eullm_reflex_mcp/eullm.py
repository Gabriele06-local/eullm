"""The EuLLM endpoints Reflex is built on, over HTTP.

  * `POST /v1/systemone`: decisions, and its errors in the System One shape,
    `{"error": {"code", "message", "question"?}}`;
  * `POST /v1/embeddings`: OpenAI-shaped embeddings;
  * `GET /v1/models` and `GET /api/version`: what the server has loaded.

Every failure becomes an `EuLLMError` that carries EuLLM's own message, so
the agent reads what EuLLM said — "question, options and readout need 2100
tokens; Jev-Style-0.8B-Decision-v3 allows 2048 — nothing was truncated" —
and not a stack trace.
"""

import json

import httpx2

from eullm_reflex_mcp import __version__

# A connection that is not made within this time will not be made: EuLLM is
# on this machine unless --allow-remote says otherwise. The answer itself
# may take minutes on a CPU, and has its own timeout (EULLM_TIMEOUT).
CONNECT_TIMEOUT = 10.0


class EuLLMError(Exception):
    """EuLLM refused a request, or could not be reached. `code` and
    `question` are those of a System One error, when EuLLM gave one."""

    def __init__(self, message, status=None, code=None, question=None):
        super().__init__(message)
        self.status, self.code, self.question = status, code, question


class EuLLM:
    """An HTTP client for one EuLLM server, opened once for the life of the
    MCP server: creating one loads the system's CA certificates, 59 ms on a
    4-core machine — more than a RAG gate decision takes on an RTX 5070 Ti
    (49 ms)."""

    def __init__(self, config):
        self.config = config
        headers = {"User-Agent": f"eullm-reflex-mcp/{__version__}"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self.client = httpx2.AsyncClient(
            base_url=config.url,
            headers=headers,
            timeout=httpx2.Timeout(config.timeout, connect=CONNECT_TIMEOUT),
            # A proxy from the environment cannot reach this machine's
            # loopback: it would send "localhost" to its own.
            trust_env=not config.loopback,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.client.aclose()

    async def systemone(self, body):
        return await self.request("POST", "/v1/systemone", body)

    async def embeddings(self, model, texts):
        """One vector per text, in the order given."""
        body = await self.request("POST", "/v1/embeddings", {"model": model, "input": texts})
        try:
            rows = sorted(body["data"], key=lambda row: row["index"])
            vectors = [row["embedding"] for row in rows]
        except (KeyError, TypeError) as e:
            raise EuLLMError(
                f"EuLLM's answer to /v1/embeddings is not the one expected: {e!r}"
            ) from None
        if len(vectors) != len(texts):
            raise EuLLMError(f"EuLLM returned {len(vectors)} embeddings for {len(texts)} inputs")
        return vectors

    async def models(self):
        return await self.request("GET", "/v1/models")

    async def version(self):
        return await self.request("GET", "/api/version")

    async def request(self, method, path, body=None):
        try:
            response = await self.client.request(method, path, json=body)
        except httpx2.TimeoutException:
            raise EuLLMError(
                f"EuLLM at {self.config.url} did not answer {path} within "
                f"{self.config.timeout:g} s (EULLM_TIMEOUT)"
            ) from None
        except httpx2.TransportError as e:
            raise EuLLMError(
                f"EuLLM is not reachable at {self.config.url}: {str(e) or type(e).__name__}. "
                "Start it with a decision model, e.g. "
                "`eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m`, "
                "or set EULLM_URL"
            ) from None
        if response.status_code >= 400:
            raise error(response, path)
        try:
            return response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EuLLMError(
                f"EuLLM answered {path} with HTTP {response.status_code} but not with JSON"
            ) from None


def error(response, path):
    """EuLLM's refusal, in its own words. `/v1/systemone` answers in the
    System One shape; the OpenAI and Ollama endpoints with
    `{"error": "message"}` or `{"error": {"message": ...}}`."""
    try:
        detail = response.json().get("error")
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        detail = None
    code = question = None
    if isinstance(detail, dict):
        code, question = detail.get("code"), detail.get("question")
        message = detail.get("message") or json.dumps(detail)
    elif isinstance(detail, str) and detail:
        message = detail
    else:
        message = response.text.strip()[:500] or response.reason_phrase
    head = f"EuLLM's {path} answered {response.status_code}" + (f" {code}" if code else "")
    where = f" (question {question!r})" if question else ""
    return EuLLMError(
        f"{head}: {message}{where}", status=response.status_code, code=code, question=question
    )
