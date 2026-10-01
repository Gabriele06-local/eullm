"""A GRPO run that watches its own signal, so nobody has to.

A whole-node job left alone overnight can spend its hours on nothing: a
reward that is NaN from the first step, or groups whose eight answers are all
right (or all wrong) step after step, which leaves no advantage to learn
from. Both show in the metrics TRL logs every step; `RunGuard` reads them,
says when to stop, and writes the line a person reads in the morning
(`status.sh` prints the last ones).

The decision depends only on logged metrics, which TRL gathers across
processes before logging, so every rank reaches the same verdict at the same
step and they stop together.
"""

from __future__ import annotations

import math
import time


class RunGuard:
    """Reads each step's logs; `observe` returns why to stop, or None.

    Two ways to stop, handled differently by the caller: ``broken`` (a NaN:
    the weights cannot be trusted, nothing is saved) and saturated (no group
    disagrees any more: what was learnt until then is kept and packaged).

    Args:
        patience: consecutive steps with almost no learning signal before
            stopping.
        dead_share: share of groups with zero reward spread that counts as
            "almost no learning signal".
        every: steps between progress lines.
        max_steps: total steps, for the progress line.
    """

    def __init__(self, patience: int = 20, dead_share: float = 0.95, every: int = 10,
                 max_steps: int = 0):
        self.patience = patience
        self.dead_share = dead_share
        self.every = every
        self.max_steps = max_steps
        self.dead_steps = 0
        self.stopped: str | None = None
        self.broken = False
        self._t0 = time.monotonic()
        self._first: int | None = None

    def observe(self, step: int, logs: dict) -> str | None:
        """Take one step's logs; the reason to stop, or None to go on."""
        if "reward" not in logs:      # a log without training metrics (e.g. the final summary)
            return None
        for key in ("loss", "reward"):
            value = logs.get(key)
            if value is not None and not math.isfinite(float(value)):
                self.broken = True
                self.stopped = f"{key} is {value} at step {step}: the run is broken"
                return self.stopped
        zero = float(logs.get("frac_reward_zero_std", 0.0))
        self.dead_steps = self.dead_steps + 1 if zero >= self.dead_share else 0
        if self.dead_steps >= self.patience:
            self.stopped = (f"{self.dead_steps} steps in a row with {zero:.0%} of groups "
                            f"all-equal (step {step}): nothing left to learn from these prompts")
            return self.stopped
        return None

    def progress(self, step: int, logs: dict) -> str | None:
        """The line to print every ``every`` steps, or None."""
        if "reward" not in logs:
            return None
        if self._first is None:      # a resumed run starts at its checkpoint's step
            self._first = step - 1
        if step % self.every:
            return None
        elapsed = time.monotonic() - self._t0
        of = f"/{self.max_steps}" if self.max_steps else ""
        return (f"[grpo] step {step}{of} reward {float(logs['reward']):.3f} "
                f"(std {float(logs.get('reward_std', 0)):.3f}) "
                f"zero-std {float(logs.get('frac_reward_zero_std', 0)):.2f} "
                f"kl {float(logs.get('kl', 0)):.4f} "
                f"len {float(logs.get('completions/mean_length', 0)):.0f} "
                f"{elapsed / max(step - self._first, 1):.0f}s/step")
