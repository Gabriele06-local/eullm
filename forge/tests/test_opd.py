"""Privileged-context distillation: the loss, the prompts, and a run on a tiny model.

The run uses a randomly initialised two-layer Qwen2 with a word-level
tokenizer saved to disk, student and teacher alike: what is pinned is the
loop around the models -- prompts read, answers sampled, the teacher behind
its own prompt, the adapter saved, a stopped run carried on, a teacher with
another vocabulary refused.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reverse_kl_is_zero_for_the_same_distribution_and_masks_positions():
    torch = pytest.importorskip("torch")
    from eullm_forge.opd import reverse_kl

    a = torch.randn(5, 11)
    assert reverse_kl(a, a).item() == pytest.approx(0.0, abs=1e-6)
    b = a.clone()
    b[2] += torch.randn(11) * 3
    assert reverse_kl(a, b).item() > 0
    mask = torch.tensor([1, 1, 0, 1, 1])
    assert reverse_kl(a, b, mask).item() == pytest.approx(0.0, abs=1e-6)
    wide = torch.cat([a, torch.full((5, 4), -1e4)], dim=1)     # padded vocabulary
    assert reverse_kl(a, wide).item() == pytest.approx(0.0, abs=1e-5)


def test_the_teacher_sees_the_source_ruling_and_the_student_does_not():
    from eullm_forge.caselaw.prompts import caselaw_prompt, privileged_prompt, ruling_label

    lab = ruling_label({"sezione": "Sezione Quinta", "numero": "202301234"})
    assert lab == "Cons. Stato, Sezione Quinta, n. 202301234"
    passages = [(lab, "Il termine decorre dalla pubblicazione." + " x" * 2000)]
    s = caselaw_prompt("Da quando decorre il termine?", passages)
    t = privileged_prompt("Da quando decorre il termine?", passages, (lab, "TESTO DELLA SENTENZA"))
    assert s.startswith("Testi di riferimento")
    assert s.endswith("Domanda: Da quando decorre il termine?")
    assert " […]" in s and "TESTO DELLA SENTENZA" not in s
    assert t.startswith("Sentenza da cui proviene la domanda") and t.endswith(s)
    hidden = privileged_prompt("Da quando decorre il termine?", passages,
                               (lab, "TESTO DELLA SENTENZA"), citable=False)
    head = hidden.split("\n", 1)[0]
    assert "202301234" not in head and "non citarla" in head
    assert "TESTO DELLA SENTENZA" in hidden and hidden.endswith(s)


def _corpus(tmp_path: Path):
    rows, cards, index = [], [], []
    for n in range(6):
        num = f"20200{n:04d}"
        text = (f"Sentenza {num}. Il ricorso riguarda materia{n}. " * 30 + "\nDIRITTO\n"
                + f"Il principio{n} si applica. " * 30)
        rows.append({"text": text, "sentence_id": f"cds/{num}", "source_id": f"cds/{num}",
                     "year": 2020, "kind": "cds", "chunk_index": 0})
        cards.append({"id": f"cds/{num}",
                      "principi": [f"Il principio{n} vale sempre in materia{n}."],
                      "norme": ["art. 120 c.p.a."], "esito": "rigetto", "materia": f"materia{n}",
                      "oggetto": "PERMESSO DI SOGGIORNO" if n == 5 else f"APPALTO {n}",
                      "domande_ricerca": [f"Quando si applica il principio{n}?",
                                          f"Cosa prevede la materia{n}?"],
                      "domande_esame": []})
        index.append(f"cds/{num}")
    chunks = tmp_path / "train.jsonl"
    chunks.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    cards_f = tmp_path / "schede.jsonl"
    cards_f.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cards))
    dev = tmp_path / "dev.txt"
    dev.write_text("cds/202000000\n")
    statutes = tmp_path / "prompts.jsonl"
    statutes.write_text("".join(json.dumps({"id": f"s{i}", "prompt": [
        {"role": "user", "content": f"Domanda sulle norme {i}"}]}) + "\n" for i in range(10)))
    return chunks, cards_f, dev, statutes


def test_prompts_leave_out_dev_and_sensitive_rulings_and_mix_statutes(tmp_path, capsys):
    chunks, cards, dev, statutes = _corpus(tmp_path)
    out = tmp_path / "opd" / "prompts.jsonl"
    mod = _load("make_opd_prompts")
    assert mod.main(["--chunks", str(chunks), "--cards", str(cards), "--dev-ids", str(dev),
                     "--statutes", str(statutes), "--mix", "0.25", "--per-ruling", "2",
                     "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    case = [r for r in rows if r["kind"] == "caselaw"]
    assert len(case) == 8                                  # 4 rulings x 2 questions
    assert {r["ruling"] for r in case} == {"cds/202000001", "cds/202000002", "cds/202000003",
                                           "cds/202000004"}
    assert sum(r["kind"] == "statute" for r in rows) == 3   # 25% of the rows
    r = case[0]
    source = r["ruling"].split("/")[1]
    assert f"n. {source}" in r["teacher"][0]["content"].split("\n", 1)[0]
    assert r["teacher"][0]["content"].endswith(r["student"][0]["content"])
    printed = capsys.readouterr().out
    assert "1 development and 1 sensitive" in printed and "principio" not in printed
    assert "source among the passages in 8)" in printed


def test_no_prompt_contains_a_development_ruling_text_anywhere(tmp_path, capsys):
    """Dev rulings are kept out of the sources but used to stay in the index.

    The chosen-source loop skips them, and the assert at the end of main only
    looks at each row's source ruling, so a dev ruling's verbatim text was
    retrieved into a train row's student prompt while the log said "left out
    1 development". The docstring promises no prompt is written from one.
    """
    mark = "DEV-MARKER-ZZZ"
    rows = [{"text": f"Sentenza {num}. Il ricorso riguarda appalto {num} {extra}",
             "sentence_id": f"cds/{num}", "source_id": f"cds/{num}",
             "year": 2020, "kind": "cds", "chunk_index": 0}
            for num, extra in (("202000001", ""), ("202000002", mark))]
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (tmp_path / "schede.jsonl").write_text("".join(json.dumps(
        {"id": f"cds/{num}", "principi": [f"Il principio vale in appalto {num}."],
         "norme": ["art. 120 c.p.a."], "esito": "rigetto", "materia": "appalto",
         "oggetto": f"APPALTO {num}",
         "domande_ricerca": [f"Quando si applica appalto {num}?"]},
        ensure_ascii=False) + "\n" for num in ("202000001", "202000002")))
    (tmp_path / "dev.txt").write_text("cds/202000002\n")
    out = tmp_path / "prompts.jsonl"
    mod = _load("make_opd_prompts")
    # -k 10 so every retrieved unit lands in the prompt: with the default 3
    # the dev chunk might simply not rank, which would prove nothing.
    assert mod.main(["--chunks", str(tmp_path / "train.jsonl"),
                     "--cards", str(tmp_path / "schede.jsonl"),
                     "--dev-ids", str(tmp_path / "dev.txt"), "-k", "10",
                     "--out", str(out)]) == 0
    prompts = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(prompts) == 1 and prompts[0]["ruling"] == "cds/202000001"
    assert mark not in json.dumps(prompts[0]["student"], ensure_ascii=False)
    assert mark not in json.dumps(prompts[0]["teacher"], ensure_ascii=False)
    assert "left out 1 development" in capsys.readouterr().out


def test_cited_numbers_reads_both_forms():
    mod = _load("cds_answer")
    text = ("Come chiarito da Cons. Stato, Sez. V, n. 202301234 e dalla sentenza n. 45/2021, "
            "nonché dall'art. 120, n. 3 c.p.a.")
    assert mod.cited_numbers(text) == {"202301234", "202100045"}
    assert mod.cited_numbers("") == set()


def test_the_exam_asks_only_development_rulings_and_checks_citations(tmp_path, monkeypatch,
                                                                     capsys):
    chunks, cards, dev, _ = _corpus(tmp_path)
    questions = tmp_path / "dev-cards.jsonl"
    questions.write_text("".join(json.dumps({"id": rid, "domande_esame": [
        {"domanda": f"Quando si applica il principio{n}?", "risposta": f"Sempre ({n}).",
         "rubrica": "Dice sempre."}] * 2}) + "\n"
        for n, rid in ((0, "cds/202000000"), (1, "cds/202000001"))))
    mod = _load("cds_answer")
    seen = []

    def fake(args, contents):
        seen.extend(contents)
        return [("Secondo n. 202000000 il principio si applica.", True),
                ("Lo dice la sentenza n. 999/2019.", False)]

    monkeypatch.setattr(mod, "generate", fake)
    out = tmp_path / "answers" / "answers-x.jsonl"
    assert mod.main(["model", "--label", "x", "--questions", str(questions), "--dev-ids",
                     str(dev), "--chunks", str(chunks), "--cards", str(cards),
                     "--answers", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["cds-202000000-0", "cds-202000000-1"]
    assert {"question", "reference", "rubric", "answer"} <= set(rows[0])
    assert all(c.startswith("Testi di riferimento") for c in seen)
    first, second = rows
    assert first["source_retrieved"] and first["context"][0] == "cds/202000000"
    assert first["cited_ok"] and first["source_cited"]
    assert not second["cited_ok"] and not second["source_cited"]
    assert first["ended"] and not second["ended"]
    printed = capsys.readouterr().out
    assert "2 questions" in printed and "principio" not in printed
    assert "cut at --max-new-tokens 0.500" in printed


def test_the_note_reaches_the_teacher_only_and_leaves_the_rows_alone():
    mod = _load("opd_train")
    teacher = [{"role": "system", "content": "s"}, {"role": "user", "content": "Domanda"},
               {"role": "assistant", "content": "a"}, {"role": "user", "content": "Ancora"}]
    noted = mod.with_note(teacher, "Rispondi in breve.")
    assert noted[3]["content"] == "Ancora\n\nRispondi in breve."
    assert noted[1]["content"] == "Domanda" and teacher[3]["content"] == "Ancora"
    assert mod.with_note(teacher, "") is teacher


@pytest.fixture
def tiny_model(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")

    words = ["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>", "user", "assistant", "Domanda:",
             "Testi", "il", "termine", "ricorso", "sentenza", "principio", "si", "applica"]
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(words)},
                                                           unk_token="[UNK]"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|im_end|>",
                                               pad_token="[PAD]")
    tok.chat_template = ("{% for m in messages %}<|im_start|> {{ m.role }} {{ m.content }} "
                         "<|im_end|> {% endfor %}<|im_start|> assistant")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(words), hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=256,
                                   eos_token_id=3, pad_token_id=1)
    model = transformers.Qwen2ForCausalLM(cfg)
    path = tmp_path / "tiny"
    model.save_pretrained(path)
    tok.save_pretrained(path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({
        "student": [{"role": "user", "content": "Domanda: il termine si applica"}],
        "teacher": [{"role": "user", "content": "sentenza principio Domanda: il termine"}]}) + "\n"
        for _ in range(4)))
    return path, prompts


def test_a_tiny_run_saves_the_adapter_and_a_stopped_run_carries_on(tiny_model, tmp_path, capsys):
    path, prompts = tiny_model
    mod = _load("opd_train")
    out = tmp_path / "run"
    base = ["--student", str(path), "--teacher", str(path), "--prompts", str(prompts),
            "--out", str(out), "--student-device", "cpu", "--teacher-devices", "cpu",
            "--batch", "2", "--max-new-tokens", "4", "--rank", "4"]
    assert mod.main(base + ["--steps", "2", "--stop-after", "1e-9"]) == 0
    assert (out / "ckpt" / "state.pt").is_file() and not (out / "adapter").exists()
    assert mod.main(base + ["--steps", "2", "--save-every", "1",
                            "--teacher-note", "Rispondi in breve.", "--anchor-statutes"]) == 0
    assert (out / "adapter" / "adapter_config.json").is_file()
    printed = capsys.readouterr().out
    assert "resuming at step 0" in printed and "step 2/2 kl" in printed
    assert "Domanda" not in printed
    assert mod.main(base + ["--steps", "2"]) == 0          # done: exits at once
    assert "nothing left to do" in capsys.readouterr().out


def test_an_empty_adapter_config_is_not_a_finished_adapter(tmp_path, capsys):
    """A 0-byte adapter_config.json is an interrupted save, not a done run.

    Existence alone used to count as done and exit 0 without even reading the
    prompts, so a killed link looked exactly like a completed one. grpo_train.py
    and stage3_sft.py test size for this reason; opd_train.py was the third
    writer of the same shape and the only one left on existence.
    """
    mod = _load("opd_train")
    out = tmp_path / "run"
    (out / "adapter").mkdir(parents=True)
    (out / "adapter" / "adapter_config.json").touch()  # the killed link's trace
    with pytest.raises(FileNotFoundError):  # it reads the prompts instead
        mod.main(["--student", "s", "--teacher", "t",
                  "--prompts", str(tmp_path / "no-such-prompts.jsonl"),
                  "--out", str(out)])
    assert "nothing left to do" not in capsys.readouterr().out


def test_every_prompts_row_is_validated_before_the_models_load(tmp_path):
    """Only the first row was checked, so a file whose later row lacks
    'teacher' passed the gate and crashed with KeyError at batch time --
    after the student and the 61 GB teacher were loaded."""
    mod = _load("opd_train")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps(r) + "\n" for r in [
        {"student": [{"role": "user", "content": "q"}],
         "teacher": [{"role": "user", "content": "t"}]},
        {"student": [{"role": "user", "content": "q"}]},
    ]))
    with pytest.raises(SystemExit, match="row 1"):
        mod.load_rows(prompts)


def test_a_teacher_with_another_vocabulary_is_refused(tiny_model, tmp_path):
    path, prompts = tiny_model
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    other = tmp_path / "other"
    import shutil
    shutil.copytree(path, other)
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "ciao": 1},
                                                           unk_token="[UNK]"))
    transformers.PreTrainedTokenizerFast(tokenizer_object=raw).save_pretrained(other)
    mod = _load("opd_train")
    with pytest.raises(SystemExit, match="tokenizers differ"):
        mod.main(["--student", str(path), "--teacher", str(other), "--prompts", str(prompts),
                  "--out", str(tmp_path / "r"), "--student-device", "cpu",
                  "--teacher-devices", "cpu"])


def test_statute_rows_are_anchored_to_the_student_before_the_run(tiny_model, tmp_path, capsys):
    """2026-10-07: the case-law teacher on statute rows cost the 4B 15:71 on statutes.

    Anchored, a statute row is taught by the student itself with its adapter
    off: on the first step the adapter is still zero, so the KL is exactly 0,
    whatever the teacher. Without the anchor the teacher -- here one whose
    output layer is scaled, so that it disagrees -- teaches it.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    path, prompts = tiny_model
    other = tmp_path / "other"
    model = transformers.AutoModelForCausalLM.from_pretrained(path)
    with torch.no_grad():
        model.lm_head.weight.mul_(50)
    model.save_pretrained(other)
    transformers.AutoTokenizer.from_pretrained(path).save_pretrained(other)
    rows = [dict(json.loads(line), kind="statute") for line in prompts.read_text().splitlines()]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    mod = _load("opd_train")
    base = ["--student", str(path), "--teacher", str(other), "--prompts", str(prompts),
            "--student-device", "cpu", "--teacher-devices", "cpu", "--batch", "2",
            "--max-new-tokens", "4", "--rank", "4", "--steps", "1"]

    def first_kl(out, *extra):
        assert mod.main(base + ["--out", str(tmp_path / out), *extra]) == 0
        line = next(ln for ln in capsys.readouterr().out.splitlines() if "step 1/1 kl" in ln)
        return float(line.split(" kl ")[1].split()[0])

    assert first_kl("anchored", "--anchor-statutes") == 0.0
    assert first_kl("taught") > 1.0
