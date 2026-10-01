"""Reinforcement learning with verifiable rewards on open-book legal questions.

Stage 3 (SFT) taught the models the open-book format; every model trained
that way stops at about four answers in five on the held-out exam. GRPO
samples several answers to the same question and pushes the model towards
the ones a checker can verify are right, so it learns from its own mistakes
rather than from a teacher's answers. Only questions whose answer can be
checked without a judge are used: see `rewards`.
"""

from __future__ import annotations

from .rewards import ABSTAIN_TYPES, DEADLINE_TYPES, abstains, answer_reward, score_answer

__all__ = ["ABSTAIN_TYPES", "DEADLINE_TYPES", "abstains", "answer_reward", "score_answer"]
