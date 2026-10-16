"""prepare_eurlex.py: EU acts in both of the Office's layouts into records.

The fixtures are made up, in the shape of the pages the Publications Office
serves (16-10-2026): the XHTML of the acts since 2014 and the plain HTML
before, with EUR-Lex's own lines in front.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_eurlex as pe  # noqa: E402

XHTML = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<p class="oj-doc-ti">REGOLAMENTO (UE) 2099/1 DEL PARLAMENTO EUROPEO</p>
<p class="oj-normal">considerando quanto segue: (1) una premessa.</p>
<div class="eli-subdivision"><p class="oj-ti-art">Articolo 1</p>
<p class="oj-sti-art">Oggetto</p>
<p class="oj-normal">1. Il presente regolamento stabilisce norme.</p>
<p class="oj-note">(1) Una nota a pie di pagina.</p></div>
<div class="eli-subdivision"><p class="oj-ti-art">Articolo 2 bis</p>
<p class="oj-normal">Una disposizione aggiunta.</p></div>
<p class="oj-doc-ti">ALLEGATO I</p><p class="oj-normal">Elenco.</p>
<p class="oj-signatory">Il presidente</p>
</body></html>"""

OLD_HTML = """<html><head><title>EUR-Lex - 31999L0001 - IT</title></head><body>
<p>Avis juridique important</p><p>31999L0001</p>
<p>Direttiva 99/1/CEE del Consiglio relativa a qualcosa Gazzetta ufficiale n. L 1 del 01/01/1999</p>
<p>*****</p><p>HA ADOTTATO LA PRESENTE DIRETTIVA:</p>
<p>Articolo 1</p><p>Gli Stati membri provvedono.</p>
<p>Articolo 2</p><p>Gli Stati membri sono destinatari.</p></body></html>"""


def test_the_xhtml_layout():
    recs = {r["article_num"]: r for r in pe.parse_act(XHTML, "32099R0001")}
    assert list(recs) == ["preambolo", "1", "2bis", "allegato_I"]
    one = recs["1"]
    assert one["article_title"] == "Oggetto"
    assert one["text"].startswith(
        "REGOLAMENTO (UE) 2099/1 DEL PARLAMENTO EUROPEO\nArticolo 1\nOggetto\n1. Il")
    assert "nota a pie" not in one["text"]
    assert "presidente" not in recs["allegato_I"]["text"]


def test_the_old_layout_without_eurlex_lines_or_publication():
    recs = {r["article_num"]: r for r in pe.parse_act(OLD_HTML, "31999L0001")}
    assert list(recs) == ["preambolo", "1", "2"]
    title = "Direttiva 99/1/CEE del Consiglio relativa a qualcosa"
    assert recs["1"]["text"] == f"{title}\nArticolo 1\nGli Stati membri provvedono."
    assert "Avis juridique" not in recs["preambolo"]["text"]
    assert "Gazzetta" not in recs["1"]["text"]


def test_a_directory_of_acts_into_one_file(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    with gzip.open(src / "32099R0001.html.gz", "wt", encoding="utf-8") as f:
        f.write(XHTML)
    (src / "31999L0001.html").write_text(OLD_HTML, encoding="utf-8")
    (src / "31999L0002.html").write_text("<html><body></body></html>", encoding="utf-8")
    assert pe.main([str(src), "--output", str(out)]) == 0
    rows = [json.loads(line) for line in (out / "legislazione_eurlex.chunks.jsonl").open()]
    assert {r["code"] for r in rows} == {"eurlex_32099R0001", "eurlex_31999L0001"}
    assert all(r["kind"] == "legge_ue" and r["text"] for r in rows)
    assert not list(out.glob("*.tmp"))
