"""Configuration, read from the environment an MCP client starts us in.

| Variable | Default | |
|---|---|---|
| `EULLM_URL` | `http://localhost:11434` | the EuLLM server |
| `EULLM_API_KEY` | none | sent as a bearer token, when EuLLM requires keys |
| `EULLM_EMBED_MODEL` | none | embedding model for `select_tools`' shortlist |
| `EULLM_EMBED_QUERY_PREFIX` | none | text put before the request when it is embedded |
| `REFLEX_GATE_THRESHOLD` | none | calibrated threshold on P(answer) for `rag_gate` |
| `EULLM_TIMEOUT` | 300 | seconds to wait for EuLLM's answer |

An empty variable counts as unset, so a client configuration can blank one
out. Only a loopback EuLLM is accepted unless the server is started with
`--allow-remote`: the requests carry the agent's text, and the roadmap keeps
remote models off unless someone turns them on.
"""

import dataclasses
import ipaddress
import os
from collections.abc import Mapping
from urllib.parse import urlsplit

DEFAULT_URL = "http://localhost:11434"

# ragbench.py's per-request timeout. On a CPU a decision takes seconds — the
# Jev-Style 2B read MetaTool's 199 tools in 76 s on a 4-core machine — and a
# model loaded on the first request adds its load time.
DEFAULT_TIMEOUT = 300.0


class ConfigError(ValueError):
    """A setting that cannot be used, said in terms of the setting."""


@dataclasses.dataclass(frozen=True)
class Config:
    url: str = DEFAULT_URL
    api_key: str | None = None
    embed_model: str | None = None
    embed_query_prefix: str = ""
    gate_threshold: float | None = None
    timeout: float = DEFAULT_TIMEOUT
    allow_remote: bool = False

    def __post_init__(self):
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConfigError(f"EULLM_URL must be an http:// or https:// URL, not {self.url!r}")
        if not self.allow_remote and not is_loopback(parts.hostname):
            raise ConfigError(
                f"EULLM_URL {self.url!r} is not on this machine: requests carry the agent's "
                "text, so only a loopback address (localhost, 127.0.0.1, ::1) is used unless "
                "the server is started with --allow-remote"
            )
        if self.gate_threshold is not None and not 0.0 <= self.gate_threshold <= 1.0:
            raise ConfigError(
                "REFLEX_GATE_THRESHOLD is a probability, between 0 and 1, "
                f"not {self.gate_threshold}"
            )
        if not self.timeout > 0:
            raise ConfigError(
                f"EULLM_TIMEOUT must be a number of seconds above 0, not {self.timeout}"
            )

    @property
    def loopback(self):
        return is_loopback(urlsplit(self.url).hostname or "")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, allow_remote=False):
        env = os.environ if env is None else env

        def get(name):
            value = env.get(name, "").strip()
            return value or None

        timeout = number("EULLM_TIMEOUT", get("EULLM_TIMEOUT"))
        prefix = env.get("EULLM_EMBED_QUERY_PREFIX", "")
        return cls(
            url=(get("EULLM_URL") or DEFAULT_URL).rstrip("/"),
            api_key=get("EULLM_API_KEY"),
            embed_model=get("EULLM_EMBED_MODEL"),
            # As ReflexBench's --embed-query-prefix: a `\n` written in a shell
            # or a JSON configuration is a newline. Qwen3-Embedding's
            # instruction ends in one before "Query:".
            embed_query_prefix=prefix.replace("\\n", "\n"),
            gate_threshold=number("REFLEX_GATE_THRESHOLD", get("REFLEX_GATE_THRESHOLD")),
            timeout=DEFAULT_TIMEOUT if timeout is None else timeout,
            allow_remote=allow_remote,
        )


def number(name, value):
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        raise ConfigError(f"{name} must be a number, not {value!r}") from None


def is_loopback(host):
    """`localhost` or a loopback address. A name other than `localhost` is
    not resolved: what it resolves to can change between this check and the
    request."""
    host = host.strip("[]").lower().rstrip(".")
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)  # ::ffff:127.0.0.1
    return address.is_loopback or bool(mapped and mapped.is_loopback)
