# decision-desk — three local models at work, on one page

Customer emails for a fictional e-bike shop arrive in six languages, and the
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

The first run on a real GPU (Jev-Style 2B, 27 October 2026) put 11 of 12
where they belong, at a median 49 ms a decision. The twelfth was the lawyer's
notice to deliver a bike: shipping 0.49, legal 0.45, and Gemma drafted a reply
to a lawyer. Every other email had its team at 0.89 or more, so legal at 0.11
or less. Hence `--legal` (0.3): from that probability of the legal team a
person answers, whichever team is first. The check run again with it: 12 of 12,
team and action, at a median 49 ms.

## A video of it

`desk_video.py` runs the inbox through your server, keeps every step of every
email with the moment it happened, and renders it with the same page, frame by
frame, in a headless browser. Three files come out, ready to post:

| File | Size | For |
|---|---|---|
| `desk-wide.mp4` | 1920×1080 | X, LinkedIn, YouTube |
| `desk-tall.mp4` | 1080×1920 | Instagram Reels, X vertical |
| `desk-readme.mp4` | 1280×720, about 1 MB | a README on GitHub |

```bash
pip install playwright
python -m playwright install chromium
sudo apt install ffmpeg          # or your system's own package
python examples/decision-desk/desk_video.py \
    --decision-model jev-style-2b-decision-v3-gguf-q4_k_m \
    --hardware "RTX 5070 Ti" \
    --caption "Jev-Style 2B decides · Gemma 4 E4B writes · Qwen3-Embedding finds"
```

With a plain `eullm serve` running, the script loads the three models before
the first email, so no load is filmed. Everything on screen is the run as it
happened: the passages, the probabilities, the rule, every token of every draft
at the moment it was written. Only the pause after each email is chosen, for
reading (`--hold`, 2.5 s): the twelve emails make a video of about 40 seconds.
The wide video shows the whole page; the tall one leaves out the inbox and the
audit trail and keeps clear of the top 200 and bottom 380 pixels and the right
edge, where Instagram lays its own text and buttons over a Reel.

The run is saved as `desk-run.json`: `--from desk-run.json` renders it again,
with another `--hold`, `--caption` or `--layouts`, without the models. ffmpeg is
run as a program of its own, the one installed on your system; nothing of it
becomes part of EuLLM. To show the README video on GitHub, drag
`desk-readme.mp4` into the comment box of any issue or pull request, copy the
`https://github.com/user-attachments/assets/…` link GitHub puts there, and
paste the link alone on a line of the README.

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
