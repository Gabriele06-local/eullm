# Corpus acquisition — sources, licences and method

What the training and evaluation corpora are made of, where each part comes
from, and how it was obtained. This describes the method; the scripts that
automate the fetching are not in this repository (see *Why the acquirers are
not here*, below).

## Sources

| Source | Content | Coverage | Licence / status |
|---|---|---|---|
| **ItalGiure** (Corte di Cassazione) | Full text of civil and criminal judgments | 2021-2026 | Judgments are public domain under art. 5 l. 633/1941 |
| **Normattiva** | Codes and the Constitution | current consolidated text | Public domain, same basis |
| **OpenGA** (Giustizia Amministrativa) | Consiglio di Stato judgment **metadata** — 17 columns, no text | 2017-2026, monthly updates | CC BY 4.0 |
| **Giustizia Amministrativa portal** | Full text of Consiglio di Stato judgments | addressed per-ruling from the OpenGA index | Public domain, same basis |

The OpenGA index is metadata only: 16.7 MB covers roughly seventy thousand
rulings for 2017-2024, about 240 bytes each. The full text lives on the
institutional portal and is addressed by two fields the index carries —
`NUMERO_RICORSO` becomes the request's `nrg` and `NUMERO_PROVVEDIMENTO` its
`nomeFile`.

## How the corpus is partitioned

**Corte di Cassazione** — the whole of 2021-2026 is training data. The
validation split is described under *Known limitations*.

**Consiglio di Stato** — split by publication year, permanently:

* **2025 and 2026 — evaluation only.** These never enter any training run.
* **2017-2024 — available for training.**

The split is temporal rather than random because it is self-verifying (the
year is a field, not a bookkeeping decision) and because it measures the
harder and more realistic thing: whether a model holds up on judgments handed
down after the ones it read.

**The 2017-2024 side is training data, and is meant to be used.** It is the
corpus of the arm in
`forge/training/configs/leonardo/distill_qwen3_30b_a3b_to_4b_cds.yaml`, which
holds every other variable identical to the split arm so that the difference
between the two is the corpus and nothing else.

**The split is enforced, not remembered.**
`forge/scripts/check_corpus_holdout.py` scans the formatted `train.jsonl` and
exits non-zero if any Consiglio di Stato record is dated 2025 or later — or
if a record from that source carries no year at all, since a record that
cannot be shown to be outside the range is not evidence that it is. The
Leonardo launcher runs it before the job does anything expensive.

This is a gate rather than a warning because the failure is silent in the
worst possible way: a contaminated corpus trains normally and reports a
*better* number, and it invalidates retroactively every transfer-curve point
already measured, since those would no longer be comparable with the ones
after. A blanket ban on recent years would be simpler and wrong — Cassazione
2021-2026 is all training data, so only the Consiglio di Stato years are
held out.

## Processing, in order

1. **Anonymisation** (`forge/eullm_forge/datasets/anonymize.py`). Judgments
   name parties, lawyers and magistrates. Persons become `[PERSONA_N]`,
   numbered consistently within a ruling so the text stays coherent;
   structured identifiers — codice fiscale, partita IVA, IBAN — are removed by
   shape. `forge/scripts/sweep_structured_pii.py` re-checks the output and
   exits non-zero when it finds anything, so it can gate a training launch.
2. **Deduplication** (`forge/eullm_forge/datasets/dedup.py`), exact then near.
3. **Chunking and formatting** (`forge/scripts/format_pretraining.py`) into
   `train.jsonl` / `val.jsonl`, split **by document** via `sentence_id`.

## The administrative vertical

legal-it is two 4B models, not one: **civile e penale** (Cassazione plus the
codes, the corpus above) and **amministrativo**. The civil/criminal model,
asked about TAR and ricorso straordinario deadlines it had never studied,
answered 30 days for both; they are 60 and 120. A 4B does one domain well, so
the administrative one gets its own corpus and its own training run
(`distill_qwen3_30b_a3b_to_4b_amm.yaml`, `sbatch_phase2_amm.slurm`):

* Consiglio di Stato **2017-2024** rulings, never 2025-2026;
* the norms they apply: *codice del processo amministrativo* (d.lgs.
  104/2010), L. 241/1990, D.P.R. 1199/1971, and the Costituzione.

The administrative norms are listed in `NORMATTIVA_LAWS_AMMINISTRATIVO` and
enter a build only when named, so the civil/criminal corpus stays as it is.

Built on the workstation, because the rulings carry personal data until they
are anonymised, in a directory of its own so nothing from the Cassazione
corpus is swept in:

```bash
C=~/work/corpus_amm; D=~/work/datasets/legal_it_amm; mkdir -p $C
# The rulings, named so the pipeline reads kind=cds from the file name.
cp ~/work/cds/cds-train.jsonl $C/italgiure_cds_2017-2024.jsonl
python forge/scripts/anonymize_italgiure.py $C --sample 20 --dry-run   # look first
python forge/scripts/anonymize_italgiure.py $C
python forge/scripts/sweep_structured_pii.py $C/italgiure_cds_*.anon.jsonl  # must exit 0
python forge/scripts/chunk_corpus.py $C
python forge/scripts/dedup_corpus.py $C
# The norms. The c.p.a. from the Normattiva codes ZIP if it is in it,
# otherwise like the other two, as a single AKN XML from normattiva.it.
python forge/scripts/prepare_legislation.py Codici_AKN_VIGENTE_*.zip --output $C \
    --sources codice_processo_amministrativo
python forge/scripts/prepare_legislation.py <l-241-1990>.xml --output $C \
    --source-id legge_procedimento_amministrativo
python forge/scripts/prepare_legislation.py <dpr-1199-1971>.xml --output $C \
    --source-id ricorsi_amministrativi
python forge/scripts/prepare_legislation.py <costituzione>.xml --output $C \
    --source-id costituzione
python forge/scripts/format_pretraining.py $C --output $D
python forge/scripts/check_corpus_holdout.py $D --files train.jsonl val.jsonl \
    --require-source cds                                                   # must exit 0
rsync -av $D/ <user>@login.leonardo.cineca.it:/leonardo_work/<project>/datasets/legal_it_amm/
```

The same `legislazione_*.chunks.jsonl` files are what the open-book
evaluation (`legal_eval.py --norms`) retrieves from.

## Known limitations

**The 2026-09 Cassazione split is chunk-level.** `format_pretraining.py`
shuffled records, and records are ~2,048-token chunks of a ruling, so one
ruling's chunks landed on both sides of the split. Any perplexity measured
against that `val.jsonl` is measured on passages whose neighbours were
trained on. Grouping by `sentence_id` fixes it for corpora built after
2026-09-21; it cannot repair the existing split, since with chunks assigned
independently at 1 % essentially no multi-chunk ruling landed wholly in
validation. The Consiglio di Stato evaluation set exists because of this.

**Anonymisation placeholders are learned.** `[PERSONA_N]` appears throughout
the training text by construction, so a model trained on it will emit the
placeholder in its own output. This is inherent to the redaction strategy, not
a defect in it, and is addressed downstream rather than in the corpus.

## Why the acquirers are not here

The scripts that fetch from the portals live in a private repository. The
politeness they apply — a delay between requests, a few hundred documents by
default, a User-Agent naming the project and a contact address — is a default
that a flag can change, and publishing that aimed at a public institution's
servers invites the load it makes easy.

The same reasoning already applied to the data: `italgiure.py` has warned for
months that the download retrieves personal data and that the raw corpus must
not be published. This extends it to the tools.

Everything needed to reproduce the *work* is here: the sources are named
above, their licences are stated, the processing is in this repository and the
evaluation corpus builder (`forge/scripts/make_ppl_corpus.py`) writes a
provenance file beside every corpus it produces. Anyone who obtains the same
public data by their own means gets the same result. That is what
reproducibility requires; an automated scraper is not.
