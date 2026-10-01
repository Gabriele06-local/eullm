"""A GRPO run stops itself when its signal is broken or gone, and says so.

A whole-node job left overnight must not spend its hours on a NaN or on
prompts the model already answers the same way eight times out of eight.
"""

from __future__ import annotations

from eullm_forge.rl.guard import RunGuard


def step_logs(reward=0.7, zero=0.3, loss=0.01, **more):
    return {"reward": reward, "reward_std": 0.3, "frac_reward_zero_std": zero,
            "loss": loss, "kl": 0.002, "completions/mean_length": 90.0, **more}


def test_a_healthy_run_goes_on_and_reports_every_ten_steps():
    guard = RunGuard(max_steps=250)
    lines = []
    for step in range(1, 31):
        assert guard.observe(step, step_logs()) is None
        if line := guard.progress(step, step_logs()):
            lines.append(line)
    assert len(lines) == 3 and lines[0].startswith("[grpo] step 10/250 reward 0.700")
    assert not guard.stopped and not guard.broken


def test_a_nan_stops_at_once_and_marks_the_run_broken():
    guard = RunGuard()
    assert guard.observe(1, step_logs()) is None
    reason = guard.observe(2, step_logs(loss=float("nan")))
    assert reason and "loss is nan" in reason and guard.broken


def test_no_signal_for_long_enough_stops_without_breaking():
    """Saturated, not broken: what was learnt is kept."""
    guard = RunGuard(patience=20)
    for step in range(1, 20):
        assert guard.observe(step, step_logs(zero=1.0)) is None
    reason = guard.observe(20, step_logs(zero=1.0))
    assert reason and "nothing left to learn" in reason and not guard.broken


def test_one_step_with_signal_resets_the_count():
    guard = RunGuard(patience=5)
    for step in range(1, 5):
        guard.observe(step, step_logs(zero=1.0))
    assert guard.observe(5, step_logs(zero=0.5)) is None
    for step in range(6, 10):
        assert guard.observe(step, step_logs(zero=1.0)) is None


def test_the_final_summary_log_is_not_a_step():
    guard = RunGuard()
    assert guard.observe(250, {"train_runtime": 1.0, "train_loss": 0.0}) is None
    assert guard.progress(250, {"train_runtime": 1.0}) is None
