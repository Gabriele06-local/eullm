"""Answer exam prompts with a GGUF file, through llama.cpp's llama-server.

What ships is a Q4_K_M GGUF, and every exam so far graded the bf16 weights
it was made from. Quantization costs something, usually little; how much on
these questions is a number, not an assumption, and it is measured by asking
the GGUF the same exam the same way:

* the prompts are built and tokenized by the HF tokenizer of the merged
  model, exactly as for the bf16 run, and sent to the server as token ids,
  so the server adds nothing (no BOS of its own, no template of its own);
* decoding is greedy, the prompt cache off, one request per prompt;
* the generated ids come back (``return_tokens``) and are decoded by the
  same tokenizer, with the same "ended its turn" rule.

The difference between the two answers files is then the quantization and
the runtime, nothing else, and compare_graded.py says whether it matters.

The server is started for one run and stopped after it. Its log goes to a
file and only its last lines are printed when it fails: they name the
problem, never a prompt, so this is safe on the held-out exam.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def default_server_binary() -> str:
    """$LCPP_SERVER, else the CUDA build next to the CPU one in $LCPP_DIR."""
    if os.environ.get("LCPP_SERVER"):
        return os.environ["LCPP_SERVER"]
    lcpp = os.environ.get("LCPP_DIR") or str(Path(os.environ.get("WORK", "~")) / "llama.cpp")
    return str(Path(lcpp).expanduser() / "build-cuda" / "bin" / "llama-server")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class LlamaServer:
    """A llama-server process on a free local port, for the length of a ``with``.

    ``parallel`` requests are decoded together; each has ``ctx_per_slot``
    tokens of context for its prompt and answer.
    """

    def __init__(self, gguf: str | Path, *, binary: str | None = None, parallel: int = 4,
                 ctx_per_slot: int = 16384, log_path: str | Path | None = None,
                 start_timeout: float = 900.0, devices: str | None = None):
        self.gguf = Path(gguf)
        self.binary = binary or default_server_binary()
        self.parallel = parallel
        self.ctx_per_slot = ctx_per_slot
        self.log_path = Path(log_path) if log_path else self.gguf.with_suffix(".server.log")
        self.start_timeout = start_timeout
        self.devices = devices          # CUDA_VISIBLE_DEVICES for this server only
        self.port = 0
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        return [self.binary, "-m", str(self.gguf), "--host", "127.0.0.1",
                "--port", str(self.port), "-ngl", "999", "-np", str(self.parallel),
                "-c", str(self.parallel * self.ctx_per_slot), "--no-webui"]

    def log_tail(self, n: int = 15) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return "(no server log)"

    def __enter__(self) -> LlamaServer:
        if not os.access(self.binary, os.X_OK):
            raise FileNotFoundError(f"no llama-server at {self.binary} -- build it with "
                                    "sbatch_build_llama_server.slurm, or set LCPP_SERVER")
        if not self.gguf.is_file():
            raise FileNotFoundError(f"no GGUF at {self.gguf}")
        self.port = _free_port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w")
        env = None
        if self.devices is not None:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.devices}
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, env=env)
        log.close()
        deadline = time.monotonic() + self.start_timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}) while "
                                   f"loading:\n{self.log_tail()}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, OSError):
                pass          # 503 while loading, refused before it listens
            if time.monotonic() > deadline:
                self.__exit__(None, None, None)
                raise TimeoutError(f"llama-server not ready after {self.start_timeout:.0f} s:"
                                   f"\n{self.log_tail()}")
            time.sleep(1)

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def complete(self, ids: list[int], max_new_tokens: int) -> dict:
        """One greedy completion of a tokenized prompt: the server's JSON reply."""
        body = {"prompt": ids, "n_predict": max_new_tokens, "temperature": 0.0,
                "top_k": 1, "cache_prompt": False, "return_tokens": True}
        try:
            return _post(self.url + "/completion", body, timeout=3600)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"llama-server refused a prompt of {len(ids)} tokens "
                               f"({e.code}): {detail}") from None


def generate_answers_gguf(server, tok, prompts: list[str], *, max_new_tokens: int,
                          end_ids: list[int], parallel: int = 4) -> list[tuple[str, bool]]:
    """(answer, ended_its_turn) per templated prompt, in order, like generate_answers.

    A turn has ended when the server stopped on an end-of-generation token
    rather than on the token budget, or when an end token is among the ids it
    returns (cut there, as the bf16 path does).
    """
    def one(prompt: str) -> tuple[str, bool]:
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r = server.complete(ids, max_new_tokens)
        if r.get("truncated"):
            raise RuntimeError(f"a prompt of {len(ids)} tokens did not fit the slot's "
                               "context: raise --server-ctx")
        out = list(r.get("tokens") or [])
        cut = next((j for j, t in enumerate(out) if t in end_ids), None)
        ended = cut is not None or r.get("stop_type") == "eos" or bool(r.get("stopped_eos"))
        if not out:           # a server that ignores return_tokens: its own text
            return (r.get("content") or "").strip(), ended
        text = tok.decode(out[:cut] if cut is not None else out, skip_special_tokens=True)
        return text.strip(), ended

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        return list(pool.map(one, prompts))
