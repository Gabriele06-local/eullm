"""The three EuLLM calls the graphs in this folder make.

  * `decide()`: `POST /v1/systemone`. Typed questions about a state, answered
    by the decision model (Reflex) with probabilities read straight from it.
    Nothing is generated, and every decision is in EuLLM's audit trail.
  * `embed()`: `POST /v1/embeddings`, the vectors retrieval ranks passages by.
  * `chat_model()`: LangChain's `ChatOpenAI`, pointed at EuLLM's
    OpenAI-compatible `/v1/chat/completions`: the chat model that writes.

LangChain has no client for `/v1/systemone`, and the embeddings are one
request, so those two are plain HTTP with the standard library.
"""

import http.client
import json
import urllib.error
import urllib.request

from langchain_openai import ChatOpenAI


class EuLLMError(RuntimeError):
    """EuLLM answered with an error, or could not be reached."""


class EuLLM:
    """One EuLLM server: its URL, the API key it requires, if it requires
    one, and how many seconds to wait for each answer."""

    def __init__(self, url="http://localhost:11434", api_key=None, timeout=120.0):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def post(self, path, payload):
        """`payload` as JSON to `path`, the key as a bearer token; the JSON
        that comes back, or `EuLLMError` saying what went wrong."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise EuLLMError(f"{path}: HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise EuLLMError(f"{self.url}: {e.reason} — is `eullm serve` running?") from None
        # Once the status line is in, the body is read outside urlopen's
        # reach, so URLError stops covering the request. What can still go
        # wrong there is the server going quiet mid-answer, a proxy answering
        # 200 with an HTML page, or a body that stops halfway -- and both
        # graphs catch EuLLMError and nothing else, so a bare TimeoutError or
        # JSONDecodeError here reaches the user as a traceback.
        except TimeoutError:
            raise EuLLMError(
                f"{path}: no answer within {self.timeout:g}s. Is the decision model "
                "loaded, and big enough for the question?"
            ) from None
        except json.JSONDecodeError as e:
            raise EuLLMError(
                f"{path}: answered 200 with something that is not JSON ({e}). "
                "Something in front of `eullm serve` answered instead?"
            ) from None
        except http.client.HTTPException as e:
            raise EuLLMError(f"{path}: the answer stopped halfway ({e})") from None

    def decide(self, state, questions, model=None):
        """One `/v1/systemone` request: every question is answered about the
        same state, which the model reads once for all of them. `state` may
        be a string or a JSON object. Returns the whole response; the
        answers are under `answers`, in the order the questions were given."""
        payload = {"state": state, "questions": questions}
        if model:
            payload["model"] = model
        return self.post("/v1/systemone", payload)

    def embed(self, texts, model):
        """One vector per text, in the order given."""
        body = self.post("/v1/embeddings", {"model": model, "input": list(texts)})
        rows = sorted(body["data"], key=lambda row: row["index"])
        return [row["embedding"] for row in rows]

    def chat_model(self, model, temperature=0.2, max_tokens=400):
        """A LangChain chat model served by EuLLM.

        The API key goes to EuLLM, never `$OPENAI_API_KEY`: the OpenAI client
        reads that variable when no key is given, and would send someone's
        OpenAI key to this server. A server without keys ignores the one
        sent here."""
        return ChatOpenAI(
            model=model,
            base_url=self.url + "/v1",
            api_key=self.api_key or "no-key",
            # Chat Completions, which EuLLM serves, never the Responses API,
            # which langchain-openai picks for some settings on its own.
            use_responses_api=False,
            temperature=temperature,
            timeout=self.timeout,
            # A local server that refused once refuses again: say so at once.
            max_retries=0,
            extra_body={
                # EuLLM's switch for a reasoning model's thinking, Qwen3's
                # among them: off, a draft or an answer from given passages
                # does not need it, and it would cost most of the time.
                "think": False,
                # langchain-openai sends a `max_tokens` setting under OpenAI's
                # newer name, `max_completion_tokens`, which EuLLM does not
                # read; this one it does.
                "max_tokens": max_tokens,
            },
        )
