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

### `decision-snake/` — a model plays Snake

```bash
python examples/decision-snake/snake_web.py   # in a browser, at http://127.0.0.1:8765
python examples/decision-snake/snake.py       # in the terminal
```

The code lists the moves that do not crash at once and computes exact facts
about each one. A move is safe when the head can still reach the tail
afterwards: a snake that can follow its own tail never traps itself. A way to
the food is safe when the snake, having eaten, can still reach its tail. The
model picks the way to the food among the safe ones, with one `choice`
question per step. The rest is arithmetic, and code does it: the only safe
move, the only safe way to the food, following the tail the long way round
while no way to the food is safe, the move with the most room when nothing is
safe. The game never waits for the model: each tick plays the answer that has
arrived by the end of the tick, and a plain rule plays when it has not.

The Jev-Style 2B is the model for this one; it reads the step counts better
than the 0.8B:

```bash
eullm pull hf.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF:Q4_K_M
eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m
```

The browser version draws the board and, for every move, what the model read,
the options it was shown with the probability it gave each, the best ones by
the facts, and how long it took; buttons pause the game, switch to the plain
rule and set the speed. The page and the game run on your machine and talk
only to your EuLLM server. On a server without a screen, forward the port with
`ssh -L 8765:127.0.0.1:8765 you@server` and open the page on your computer.

On a GPU a decision takes a few tens of milliseconds and the game runs in real
time. On a CPU a decision takes a second or two: tick "wait for every answer"
on the page, or add `--tick 0` in the terminal.

What mattered, all found by measuring:

- **A safe move is one that keeps the tail within reach.** The first version
  judged safety by the room left and ate wherever the food was: long snakes
  coiled up and walled themselves in, at about 55 points out of the 189 a
  16×12 board holds. Keeping the tail within reach, and checking before
  eating that it stays so, took the plain rule to 186 on average over five
  games, one of them with the board full.
- **The model gets the choices it can make.** Asked which move follows the
  tail the longest way round, the 0.8B picked a right one 15 times out of 30:
  luck. Offered a way to the food that points at it but ends in a trap, it
  took it 20 times out of 20, whatever the facts said. Earlier, offered a
  dead end next to a risky move with ten times the room, it took the dead
  end. So the model is asked only which safe way to the food to take.
- **Nothing in the state that the model could follow instead of the facts.**
  With the snake's heading in the state, the 0.8B tended to carry straight
  on. With the food's direction in it, both models took the move pointing at
  the food even where the shortest safe way starts the other way round: on 20
  such boards the 0.8B took the shortest way once and the 2B 11 times.
  Without it, the 2B took it 16 times, and on 20 boards out of 20 where the
  shortest way points at the food; the 0.8B 9 and 17 times.

```bash
# no display: ten games, then the rule alone on the same boards, to compare
python examples/decision-snake/snake.py --headless --games 10 --seed 1
python examples/decision-snake/snake.py --headless --games 10 --seed 1 --player rule
# every move the model made, with the options it was shown
python examples/decision-snake/snake.py --headless --log moves.jsonl
```

#### A video of it

`decision-snake/snake_video.py` plays a game with the model and renders it,
frame by frame, with the same page in a headless browser. Three files come
out, ready to post:

| File | Size | For |
|---|---|---|
| `snake-wide.mp4` | 1920×1080 | X |
| `snake-tall.mp4` | 1080×1920 | Instagram Reels, X vertical |
| `snake-readme.mp4` | 1280×720, a few MB | a README on GitHub |

```bash
pip install playwright
python -m playwright install chromium
sudo apt install ffmpeg          # or your system's own package
python examples/decision-snake/snake_video.py --url http://localhost:11434 \
    --games 3 --caption "Jev-Style 2B on an RTX 5070 Ti"
```

Every move on screen is the game as it was played: the model's answers, each
with the time it took, and the moves code made. Only the pace is chosen for
watching: the opening moves at five a second, then fast forward, marked on
screen, then the end of the game. `--games 3` keeps the
best of three games; `--seconds` sets the length, 60 by default (X takes up
to 140 without a subscription). The game is saved as `snake-game.jsonl`, and
`--from snake-game.jsonl` renders it again without playing. ffmpeg is run as
a program of its own, the one installed on your system; nothing of it
becomes part of EuLLM. In the vertical video the score, the board and the
move stay clear of the text and buttons Instagram lays over a Reel.

To show the video in a README on GitHub, drag `snake-readme.mp4` into the
comment box of any issue or pull request (it need not be posted), copy the
`https://github.com/user-attachments/assets/…` link GitHub puts there, and
paste the link alone on a line of the README: GitHub shows it as a player.
On a free plan a video can be at most 10 MB.

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
