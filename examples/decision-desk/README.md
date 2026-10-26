# decision-desk — three local models at work, on one page

Customer emails for a fictional e-bike shop arrive in five languages, and the
page shows every one going through four steps, as they happen:

| Step | Model | What you see |
|---|---|---|
| **1 · Find** | Qwen3-Embedding 0.6B (`/v1/embeddings`) | the shop's three policy passages closest to the email, with their similarity |
| **2 · Decide** | Jev-Style 2B (`/v1/systemone`) | five questions answered at once, as probability bars: which team, legitimate or phishing or spam, needs an answer today, how upset, talks about health. Nothing is generated; it takes tens of milliseconds on a GPU |
| **3 · Rule** | none: plain code | what happens: quarantined, spam, to a person, or a reply by a team, with the rule that decided and what code found by checksum (IBAN, card number, tax code) |
| **4 · Write** | Gemma 4 E4B (`/v1/chat/completions`, streamed) | the reply drafted in the customer's language from the passages of step 1, or **0 tokens** when the rule said no draft |

Above it, the totals: emails handled, median decision time, drafts written,
generations avoided, tokens written. Below it, the lines EuLLM wrote to its
audit trail for that email, the email itself only as a SHA-256.

"Run the inbox" goes through the twelve built-in emails one after the other;
"Write one" sends your own, in any language. The page and the models run on
your machine, and nothing else is involved.

## Start it

One command on Linux or macOS. It installs EuLLM if it is missing, pulls the
three models (about 7 GB the first time), starts the server and opens the
page at http://127.0.0.1:8770:

```bash
curl -fsSL https://raw.githubusercontent.com/eullm/eullm/main/examples/decision-desk/start.sh | sh
```

From a clone of the repository, the same: `sh examples/decision-desk/start.sh`.
Ctrl+C stops the page and the server the script started. Only `python3` (3.8
or later) is needed beyond EuLLM: the desk uses the standard library alone.

| Variable | Default | |
|---|---|---|
| `DESK_SMALL=1` | off | the Jev-Style 0.8B instead of the 2B: the three models then fit an 8 GB GPU more easily |
| `DESK_CHAT` | `gemma-4-e4b` | another chat model, any you have pulled |
| `DESK_PORT` | 8770 | the page's port |
| `EULLM_PORT` | 11434 | the server's port |

By hand, or on Windows, with EuLLM installed:

```bash
eullm pull hf.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF:Q4_K_M
eullm pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
eullm pull gemma-4-e4b
eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m \
            --embedding-model qwen3-embedding-0.6b-gguf-q8_0 \
            --default-model gemma-4-e4b
python examples/decision-desk/desk.py      # in another terminal
```

The decision and embedding models load at startup and stay resident; the desk
loads Gemma 4 as soon as it starts, so the first email does not wait for it.
The three dots at the top of the page turn green when each model is loaded.
On a server without a screen, forward the port with
`ssh -L 8770:127.0.0.1:8770 you@server` and open the page on your computer:
`start.sh` then opens no browser and prints that command; `desk.py` by hand
needs `--no-browser`.

## Measure before you show it

```bash
python examples/decision-desk/desk.py --check
```

decides the twelve built-in emails without the page and without drafting, and
prints where each went against where it should have gone, then the totals and
the median decision time. Each email carries its expected team and action in
`SAMPLES`. The questions are asked in English about emails in German, Italian,
French, Spanish, Dutch and English, and the thresholds (`--quarantine`,
`--threshold`, `--min-team`) are the ones measured on Italian emails in
`decision-triage/`: run the check on your own hardware and model before a demo,
and change the samples, the questions or the thresholds where it fails. The
rules are in `policy()`, a dozen lines of plain code.

## Make it yours

Everything the demo knows is at the top of `desk.py`: `TEAMS` (the teams and
what each handles), `KNOWLEDGE` (the policy passages the embedding model
searches and Gemma answers from), `SAMPLES` (the inbox and the labels the check
compares to) and `questions()`. The page reads the teams and the inbox from the
desk, so nothing in `desk.html` needs changing.

## Licences

EuLLM is AGPL-3.0-or-later; the desk is standard-library Python and one HTML
page with no external scripts. The models are downloaded by `eullm pull` from
their publishers on Hugging Face, under their own licences: check each model
card before you redistribute anything built on them.

## Tests

```bash
python -m unittest discover -s examples/decision-desk -v
```

They run the desk against a fake EuLLM on 127.0.0.1 that answers with the
labels: the order of the steps, the rules, that the chat model is never called
for an email set aside, the page's two endpoints, the audit lines read back.
They check the plumbing, not a model; `--check` against a real server does
that.
