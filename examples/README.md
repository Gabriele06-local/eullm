# examples/

Small programs that use EuLLM the way an application would. Python, standard
library only: copy one and change it.

## Decisions with `/v1/systemone`

Both examples ask a decision model typed questions and act on the
probabilities it answers with — nothing is generated. Start the server with a
decision model first; a Jev-Style model is the one trained for this:

```bash
eullm pull hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M
eullm serve --decision-model jev-style-0.8b-decision-v3-gguf-q4_k_m
```

`--url` points either script at another server, and `--api-key` (or
`EULLM_API_KEY`) sends a key when the server requires one.

### `decision-snake/snake.py` — a model plays Snake

```bash
python examples/decision-snake/snake.py
```

The code lists the moves that do not crash at once and computes exact facts
about each one: how far the food is afterwards, how much room is left, whether
the snake can still reach its own tail. The model reads the facts and picks a
move with one `choice` question per step; a move the code knows to be a dead
end is not offered while a safe one exists. The game never waits for it: each
tick plays the answer that has arrived by the end of the tick, and a plain rule
plays when it has not.

On a GPU a decision takes a few tens of milliseconds and the game runs in real
time. On a CPU a decision takes a second or two: add `--tick 0` to wait for
every answer.

The one thing that mattered in the prompt: the state says where the food is,
but not which way the snake is heading. With the heading named, the Jev-Style
0.8B tended to carry straight on even when the facts said otherwise; without
it, it picked one of the best moves by the facts 60 times out of 60 in a game,
and on 40 boards out of 40.

```bash
# no display: ten games, then the rule alone on the same boards, to compare
python examples/decision-snake/snake.py --headless --games 10 --seed 1
python examples/decision-snake/snake.py --headless --games 10 --seed 1 --player rule
# every move the model made, with the options it was shown
python examples/decision-snake/snake.py --headless --log moves.jsonl
```

### `decision-triage/triage.py` — sorting incoming email

```bash
python examples/decision-triage/triage.py                          # built-in samples
python examples/decision-triage/triage.py --dir inbox/ --csv triage.csv
python examples/decision-triage/triage.py --mbox archive.mbox --teams teams.json
```

Each email is one request with five questions, and the email is read once for
all of them: which team handles it, is it legitimate, phishing or spam, does it
need an answer today, how upset is the sender, does it talk about someone's
health. An IBAN, a payment card number or a tax code is not asked about: code
finds it by pattern and checksum. What happens next is plain code in
`policy()`: phishing and spam are set aside, an email whose team the model is
unsure of goes to a person, urgency or anger raises the priority, personal
data is flagged. With the Jev-Style 0.8B, all 13 built-in samples go where
they should — the two phishing emails to quarantine, the genuine payment
reminder next to them to accounting.

How a question is put matters as much as the model. "Is this phishing?" as a
yes/no question put an angry customer, a lawyer's letter and a job application
over 0.7; the same question as a choice between legitimate, phishing and spam
kept every legitimate email under 0.5 and both phishing emails over 0.8. Try a
question on a few emails you know the answer to before trusting it.

`--teams` takes a JSON object of team name → what the team handles, to describe
your own organisation:

```json
{
  "amministrazione": "fatture, pagamenti, rimborsi",
  "assistenza": "problemi tecnici, accesso, errori",
  "commerciale": "preventivi e offerte"
}
```

Nothing leaves the server, and every decision is in its audit trail, the email
itself only as a SHA-256.
