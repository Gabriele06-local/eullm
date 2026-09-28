"""The exam round, run as the watcher runs it, with sbatch stubbed.

Drawing the exam is real (make_norm_exam.py on made-up articles); only the
scheduler is fake. What is pinned: the exam leaves out the training pairs'
articles, an existing exam is reused rather than redrawn, the judge waits on
the exam (afterok), and a round without its exclusions refuses to start.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "forge" / "scripts" / "leonardo" / "sbatch_exam_round.slurm"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

FILLER = " Il presente articolo contiene disposizioni di dettaglio sufficienti." * 3
RECORDS = [{"code": "codice_civile", "article_num": "", "chunk_index": 0,
            "text": f"Art. {n}. \n \n (Rubrica {n}). \n \n La domanda si propone entro "
                    f"sessanta giorni.{FILLER}"}
           for n in range(1, 61)]


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def leonardo(tmp_path):
    work = tmp_path / "work"
    (work / "norms").mkdir(parents=True)
    (work / "norms" / "legislazione_x.chunks.jsonl").write_text(
        "\n".join(json.dumps(r) for r in RECORDS) + "\n")
    model = work / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    pairs = work / "openbook-pairs.jsonl"
    pairs.write_text("\n".join(json.dumps({"key": f"ob-g-codice_civile-{n}"})
                               for n in range(1, 21)) + "\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "sbatch", '#!/usr/bin/env bash\necho "$*" >> "$CALLS"\n'
                          'echo $((100 + $(wc -l < "$CALLS")))\n')
    _exe(bin_ / "python", f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
    calls = tmp_path / "sbatch-calls"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(work),
           "CALLS": str(calls), "ROUND_REPO": str(REPO), "EULLM_VENV": str(tmp_path / "nov"),
           "EULLM_RUN_DIR": str(tmp_path / "run"),
           "ROUND_MODELS": f"v0.3={model} base={model}", "ROUND_PAIRS": str(pairs)}
    env.pop("SLURM_JOB_ID", None)

    def run(**extra):
        r = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env={**env, **extra},
                           capture_output=True, text=True, timeout=120)
        return r, calls.read_text() if calls.exists() else ""

    return run, work


def test_a_round_draws_without_trained_articles_then_exam_then_judge(leonardo):
    run, work = leonardo
    r, calls = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "20 trained articles left out" in r.stdout
    drawn = [json.loads(x)["metadata"] for x in
             (work / "eval" / "norm-exam-v2.jsonl").read_text().splitlines()]
    assert drawn
    assert not {int(m["articolo"]) for m in drawn if m["tipo"] != "inesistente"} & set(range(1, 21))
    exam, judge = calls.splitlines()
    assert "sbatch_norm_exam.slurm" in exam
    assert "--dependency=afterok:101" in judge and "sbatch_judge.slurm" in judge
    assert "Che cosa prevede" not in r.stdout   # counts only


def test_an_existing_exam_is_reused_not_redrawn(leonardo):
    run, work = leonardo
    run()
    exam = work / "eval" / "norm-exam-v2.jsonl"
    before = exam.read_text()
    r, _ = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "reusing it, not redrawing" in r.stdout
    assert exam.read_text() == before


def test_a_round_without_its_exclusions_refuses(leonardo):
    run, _ = leonardo
    r, calls = run(ROUND_PAIRS="")
    assert r.returncode == 1
    assert "ROUND_PAIRS is required" in r.stderr
    assert not calls


def test_a_development_round_shares_no_article_with_the_held_out_exam(leonardo):
    run, work = leonardo
    run()                                         # the held-out exam
    held = work / "eval" / "norm-exam-v2.jsonl"
    dev = work / "eval" / "norm-exam-dev.jsonl"
    r, calls = run(ROUND_ITEMS=str(dev), ROUND_OUT=str(work / "eval" / "dev"),
                   ROUND_EXCLUDE_EXAM=str(held), ROUND_QUIET="0")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "articles of the other exam left out" in r.stdout

    def arts(p):
        return {(json.loads(x)["metadata"]["code"], json.loads(x)["metadata"]["articolo"])
                for x in p.read_text().splitlines()}
    assert any(json.loads(x)["metadata"]["tipo"] == "contenuto"
               for x in dev.read_text().splitlines())
    assert not arts(dev) & arts(held)
