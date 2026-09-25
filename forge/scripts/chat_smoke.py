#!/usr/bin/env python3
"""Does a stage-3 model answer, and does it END its turn? Asked of the model itself.

legal-it-4b v0.1 answered its first question correctly and then never
stopped: after "Mi chiamo EULLM Legal IT." it wrote "Intialized" instead of
<|im_end|>, and went on for four thousand tokens. Perplexity did not notice
— it scored the model like the distilled checkpoint under it — and nobody
did until a person typed into the chat. This asks, every time a model is
packaged, the two things perplexity cannot:

  * does it close its turn (the last generated token is the end-of-turn
    token, well before the length limit)?
  * what does it say? The answers are printed so a human can read three of
    them in the log, not so a script can grade them.

Greedy decoding on the merged HF directory — before quantization, so a
failure here is the model's and not the GGUF's. CPU is enough: three short
generations of a 4 B model take a few minutes.

    python forge/scripts/chat_smoke.py <merged-dir> [--csv smoke.csv --label NAME]

Exit status is 0 whatever the verdict: this reports on a model, it does not
fail the job that built it. The verdict is the last line and the CSV row.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

QUESTIONS = [
    "Come ti chiami?",
    "Se ho un contenzioso con lo Stato, come posso fare ricorso?",
    "Che cosa prevede l'articolo 2043 del codice civile?",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="merged HF directory")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--csv", help="append one row per model here")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    end_ids = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
               if isinstance(i, int) and i >= 0]

    stopped = 0
    lengths = []
    for q in QUESTIONS:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
        )
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, eos_token_id=end_ids)
        new = out[0, ids["input_ids"].shape[1]:]
        ended = len(new) > 0 and int(new[-1]) in end_ids
        stopped += ended
        lengths.append(len(new))
        print(f"\n[smoke] Q: {q}")
        print(tok.decode(new, skip_special_tokens=False).strip())
        print(f"[smoke] {len(new)} tokens, {time.time() - t0:.0f}s, "
              f"ended its turn: {'YES' if ended else 'NO'}", flush=True)

    verdict = "OK" if stopped == len(QUESTIONS) else "DOES NOT STOP"
    print(f"\n[smoke] {stopped}/{len(QUESTIONS)} answers ended their turn — {verdict}")
    if args.csv:
        path = Path(args.csv)
        new_file = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "label", "model", "ended", "asked", "tokens"])
            w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), args.label, args.model,
                        stopped, len(QUESTIONS), " ".join(map(str, lengths))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
