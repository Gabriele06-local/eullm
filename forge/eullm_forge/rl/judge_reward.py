"""A judge model's grade as a GRPO reward, for the questions no program can check.

The verifiable rewards (`rewards`) cover deadlines and absent articles, and
on those the models have run out of things to learn: in the second round
(2026-10-04) three chains of four stopped early because every answer of a
group scored the same. What the held-out exams still find wrong is mostly
"what does this article provide", which has no number to check. Here a
judge grades each answer against the article, with the exam's own grading
prompt and version-2 rubric (`ReferenceGrader`, `rubric_v2`): correct 1,
partial 0.5, wrong 0.

The judge is reached over HTTP, an OpenAI-compatible chat endpoint
(llama-server, see sbatch_grpo.slurm's GRPO_JUDGE_GGUF), so it runs on its
own GPU beside the policy instead of inside the training processes.

Training against a judge invites learning what pleases the judge. A result
of a run with this reward counts only if it also holds under a second judge
the run never saw (Qwen3.6-27B on the development set).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..eval.judge import GRADE_SCORES, ReferenceGrader
from .rewards import JUDGED_TYPES, _text


def _chat(url: str, prompt: str, max_tokens: int, timeout: float) -> str:
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "max_tokens": max_tokens}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


class JudgeReward:
    """TRL reward function: the judge's grade of each judged completion.

    Completions of other kinds get None (the verifiable reward scores them).
    A reply the grader cannot read is asked again once; if it is still
    unreadable, or the judge cannot be reached, the completion gets None and
    is counted in ``unscored`` rather than given a zero it did not earn.
    """

    __name__ = "judge_reward"      # TRL names the reward's log column after it

    def __init__(self, url: str, *, parallel: int = 8, max_tokens: int = 120,
                 timeout: float = 300.0):
        self.url = url
        self.parallel = parallel
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.unscored = 0

    def grade(self, question: str, reference: str, answer: str, rubric: str) -> float | None:
        prompt = ReferenceGrader(lambda p: p).prompt(question, reference, answer, rubric)
        for _ in range(2):
            try:
                raw = _chat(self.url, prompt, self.max_tokens, self.timeout)
            except (urllib.error.URLError, OSError, KeyError, ValueError):
                continue
            label = ReferenceGrader.parse(raw).label
            if label in GRADE_SCORES:
                return GRADE_SCORES[label]
        self.unscored += 1
        return None

    def __call__(self, completions, tipo, question, reference, rubric, **_) -> list[float | None]:
        jobs = [(i, q, ref, _text(c), rub)
                for i, (c, t, q, ref, rub) in enumerate(zip(completions, tipo, question,
                                                            reference, rubric))
                if t in JUDGED_TYPES]
        out: list[float | None] = [None] * len(completions)
        with ThreadPoolExecutor(max_workers=max(1, self.parallel)) as pool:
            for (i, *_rest), score in zip(jobs, pool.map(lambda j: self.grade(*j[1:]), jobs)):
                out[i] = score
        return out
