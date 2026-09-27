#!/usr/bin/env python3
"""Shared-prefix benchmark for the decision layer (`POST /v1/systemone`).

Asks Q questions about the same state, for every Q in --questions and every
state size in --states, twice: once in `shared_prefix` mode (the state is
decoded once and shared by every question) and once in `separate` mode (each
question decoded on its own, the baseline). For each pair it reports:

  * tokens decoded by each mode, and the ratio — the saving the shared prefix
    is supposed to buy, measured from the server's own counts;
  * decode time of each mode (prefix + questions, from the server's
    `eullm.timings_ms`, median of --repeat runs) and the speedup;
  * how far the two modes' answers are apart: the largest difference in any
    raw probability and in any log-probability (codes the model all but rules
    out, below -15, are left out), and whether every argmax agrees.

The two modes read the same tokens, so on an F32 model they agree to ~1e-6;
on quantized weights they differ by the model's own numerical noise, the
amount a prompt moves when decoded in batches of a different shape (see the
equivalence test in engine/src/inference/decision.rs). That difference is
what the last columns show — it is a property of the model, not a bug, but
it is the number to know before comparing calibrations measured in one mode
with answers served in the other.

Start the server with a decision model and enough context for the largest
case (a 4k-token state with 64 questions needs about 9k tokens):

    eullm serve --decision-model qwen3-0.6b --decision-ctx 16384

then:

    python bench/decision_bench.py --url http://localhost:11434
    python bench/decision_bench.py --states 256,1024 --questions 1,8,64 --repeat 5 \\
        --json results/decision-bench.json

--details prints, for every case, the question whose answer moved most, with
its coverage in both modes; --order-check asks the same questions a third
time in reverse order, to measure how much an answer depends only on where
its question sits in the batch.

Only the Python standard library is needed. Exit code is nonzero if any
request failed for a reason other than exceeding --decision-ctx.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# Sentences for a synthetic support/legal ticket history. Varied enough that
# the state does not collapse into a few repeated tokens.
SENTENCES = [
    "The customer reports that payouts to the connected bank account have failed since Monday.",
    "A second charge for the September invoice appeared on the card statement on the 14th.",
    "Support replied with the standard refund policy and asked for the transaction reference.",
    "The client's lawyer wrote that the supplier has not delivered the goods for three months.",
    "A hearing before the court of Milan is scheduled for Thursday at ten in the morning.",
    "The account was migrated to the new billing platform during the weekend maintenance window.",
    "Two other merchants in the same region reported delayed settlements on the status page.",
    "The contract includes a penalty clause of two percent of the order value per week of delay.",
    "The customer asked to be called back today and mentioned that payroll is due on Friday.",
    "An internal note says the risk team placed a temporary hold on the account on Tuesday.",
    "The invoice lists VAT at twenty-two percent and a reverse charge note for the EU shipment.",
    "The supplier claims that force majeure applies because of a strike at the port of Genoa.",
]

TOPICS = [
    "a failed payout",
    "a duplicate charge",
    "a court hearing",
    "a contract penalty",
    "a platform migration",
    "an account hold",
    "a VAT question",
    "a delivery delay",
    "a callback request",
    "a refund request",
    "a strike",
    "a legal deadline",
    "a status page incident",
    "payroll",
    "an invoice dispute",
    "a bank transfer",
]


def make_state(target_tokens):
    """About `target_tokens` tokens of ticket history (~4 characters per
    token for English text; the server reports the real count)."""
    target_chars = target_tokens * 4
    lines = []
    length = 0
    while length < target_chars:
        i = len(lines)
        line = f"[day {1 + i // len(SENTENCES)}] {SENTENCES[i % len(SENTENCES)]}"
        lines.append(line)
        length += len(line) + 1
    return "\n".join(lines)


def make_questions(n):
    """`n` distinct questions cycling through the three types, so every
    question's own tokens differ from the others' — as real ones do."""
    questions = {}
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        round_ = i // len(TOPICS)
        kind = ("noul", "choice", "score")[i % 3]
        qid = f"q{i:02d}"
        if kind == "noul":
            questions[qid] = {
                "type": "noul",
                "instructions": f"Does the state mention {topic}? (check {round_ + 1})",
            }
        elif kind == "choice":
            questions[qid] = {
                "type": "choice",
                "instructions": f"Which team should handle {topic}? (check {round_ + 1})",
                "criteria": {
                    "billing": "Charges, invoices, refunds and payouts",
                    "legal": "Contracts, disputes and court deadlines",
                    "tech": "Bugs, outages and migrations",
                    "other": "Anything else",
                },
            }
        else:
            questions[qid] = {
                "type": "score",
                "instructions": f"How urgent is {topic} for the customer? (check {round_ + 1})",
                "criteria": [
                    "Not urgent",
                    "Can wait a week",
                    "Should be handled this week",
                    "Should be handled today",
                    "Needs an immediate response",
                ],
            }
    return questions


class RequestFailed(Exception):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RequestFailed(e.code, e.read().decode(errors="replace")) from None


def decode_ms(response):
    t = response["eullm"]["timings_ms"]
    return t["prefix"] + t["questions"]


def compare(shared, separate):
    """Largest raw-probability and log-probability difference between the
    two modes' answers, and whether every argmax agrees."""
    max_dp = 0.0
    max_dlp = 0.0
    agree = True
    for qid, a in shared["answers"].items():
        b = separate["answers"][qid]
        pa, pb = a["eullm"]["raw_probabilities"], b["eullm"]["raw_probabilities"]
        la, lb = a["eullm"]["logprobs"], b["eullm"]["logprobs"]
        for label in pa:
            max_dp = max(max_dp, abs(pa[label] - pb[label]))
            if max(la[label], lb[label]) > -15.0:
                max_dlp = max(max_dlp, abs(la[label] - lb[label]))
        if max(pa, key=pa.get) != max(pb, key=pb.get):
            agree = False
    return max_dp, max_dlp, agree


def worst(shared, other):
    """The question whose raw probabilities differ most between two runs:
    `(id, largest probability difference, answer in shared, answer in other)`."""
    found = None
    for qid, a in shared["answers"].items():
        b = other["answers"][qid]
        pa, pb = a["eullm"]["raw_probabilities"], b["eullm"]["raw_probabilities"]
        dp = max(abs(pa[k] - pb[k]) for k in pa)
        if found is None or dp > found[1]:
            found = (qid, dp, a, b)
    return found


def show_worst(shared, separate, request_ms, context_ms):
    """Print the question that moved most, with its coverage in both modes:
    similar coverage and a shifted distribution is arithmetic; a coverage
    that collapses in one mode would mean that mode read the wrong logits."""
    qid, dp, a, b = worst(shared, separate)

    def fmt(p):
        return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in p.items()) + "}"

    print(
        f"{'':>11}worst {qid} ({a['type']}): dP {dp:.3f}, coverage "
        f"{a['eullm']['coverage']:.4f} shared / {b['eullm']['coverage']:.4f} separate; "
        f"shared request {request_ms:.1f} ms (context {context_ms:.1f} ms)"
    )
    print(f"{'':>13}shared   {fmt(a['eullm']['raw_probabilities'])}")
    print(f"{'':>13}separate {fmt(b['eullm']['raw_probabilities'])}")


def run_case(url, model, state, questions, mode, repeat, timeout):
    payload = {"state": state, "questions": questions, "eullm": {"mode": mode}}
    if model:
        payload["model"] = model
    runs = []
    for _ in range(repeat):
        started = time.perf_counter()
        response = post(url, payload, timeout)
        response["_client_ms"] = (time.perf_counter() - started) * 1000.0
        runs.append(response)
    return runs


def median(values):
    return statistics.median(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--url", default="http://localhost:11434", help="server base URL"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="decision model to use (default: the one the server has loaded)",
    )
    parser.add_argument(
        "--states",
        default="256,1024,4096",
        help="approximate state sizes in tokens, comma-separated",
    )
    parser.add_argument(
        "--questions",
        default="1,4,8,16,32,64",
        help="questions per request, comma-separated (max 64)",
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="runs per case; medians are reported"
    )
    parser.add_argument(
        "--max-separate-tokens",
        type=int,
        default=150_000,
        help="skip the separate baseline when it would decode more tokens than this",
    )
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="per-request timeout, seconds"
    )
    parser.add_argument(
        "--json", default=None, help="write every case's numbers to this file"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="print the question that differs most between the modes, per case",
    )
    parser.add_argument(
        "--order-check",
        action="store_true",
        help="also ask the questions in reverse order (shared prefix both times) "
        "and report how far the answers move",
    )
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/v1/systemone"
    states = [int(s) for s in args.states.split(",") if s.strip()]
    counts = [int(q) for q in args.questions.split(",") if q.strip()]
    if any(q < 1 or q > 64 for q in counts):
        parser.error("--questions must be between 1 and 64")

    # Warm-up: loads the model if --model names one, and keeps first-request
    # costs out of the first case.
    try:
        warm = run_case(
            url,
            args.model,
            make_state(64),
            make_questions(2),
            "shared_prefix",
            1,
            args.timeout,
        )[0]
    except (RequestFailed, OSError) as e:
        print(f"Warm-up request failed: {e}", file=sys.stderr)
        return 1
    # Which attention kernels the server runs decisions with: the numbers
    # below depend on it, and a comparison of two runs means nothing if
    # both turn out to have used the same setting.
    flash_attn = warm["eullm"].get("flash_attn", "unknown")
    print(f"model {warm['model']}, flash attention {flash_attn}\n")

    header = (
        f"{'state':>6} {'Q':>3} | {'tokens sh/sep':>15} {'saving':>7} | "
        f"{'decode ms sh':>12} {'sep':>10} {'speedup':>8} | "
        f"{'max dP':>8} {'max dlogP':>9} {'argmax':>6}"
    )
    print(header)
    print("-" * len(header))

    results = []
    failed = False
    for state_tokens in states:
        state = make_state(state_tokens)
        for q in counts:
            questions = make_questions(q)
            case = {"state_tokens_target": state_tokens, "questions": q}
            try:
                shared = run_case(
                    url,
                    args.model,
                    state,
                    questions,
                    "shared_prefix",
                    args.repeat,
                    args.timeout,
                )
            except RequestFailed as e:
                if e.status == 400 and "--decision-ctx" in e.body:
                    print(f"{state_tokens:>6} {q:>3} | exceeds --decision-ctx, skipped")
                    case["skipped"] = "exceeds --decision-ctx"
                    results.append(case)
                    continue
                print(
                    f"{state_tokens:>6} {q:>3} | shared_prefix failed: {e}",
                    file=sys.stderr,
                )
                failed = True
                continue
            s0 = shared[0]["eullm"]
            case.update(
                {
                    "prompt_tokens": s0["prompt_tokens"],
                    "shared_prefix_tokens": s0["shared_prefix_tokens"],
                    "shared_evaluated_tokens": s0["evaluated_tokens"],
                    "shared_decode_ms": median([decode_ms(r) for r in shared]),
                    "shared_request_ms": median(
                        [r["eullm"]["request_ms"] for r in shared]
                    ),
                    "shared_readout_ms": median(
                        [r["eullm"]["timings_ms"]["readout"] for r in shared]
                    ),
                    "shared_context_ms": median(
                        [r["eullm"]["timings_ms"]["context"] for r in shared]
                    ),
                }
            )

            separate = None
            if s0["prompt_tokens"] <= args.max_separate_tokens:
                try:
                    separate = run_case(
                        url,
                        args.model,
                        state,
                        questions,
                        "separate",
                        args.repeat,
                        args.timeout,
                    )
                except RequestFailed as e:
                    print(
                        f"{state_tokens:>6} {q:>3} | separate failed: {e}",
                        file=sys.stderr,
                    )
                    failed = True
            if separate:
                sep0 = separate[0]["eullm"]
                max_dp, max_dlp, agree = compare(shared[0], separate[0])
                case.update(
                    {
                        "separate_evaluated_tokens": sep0["evaluated_tokens"],
                        "separate_decode_ms": median([decode_ms(r) for r in separate]),
                        "separate_request_ms": median(
                            [r["eullm"]["request_ms"] for r in separate]
                        ),
                        "max_probability_difference": max_dp,
                        "max_logprob_difference": max_dlp,
                        "argmax_agrees": agree,
                    }
                )
                saving = sep0["evaluated_tokens"] / s0["evaluated_tokens"]
                speedup = case["separate_decode_ms"] / case["shared_decode_ms"]
                print(
                    f"{state_tokens:>6} {q:>3} | "
                    f"{s0['evaluated_tokens']:>6}/{sep0['evaluated_tokens']:<8} {saving:>6.1f}x | "
                    f"{case['shared_decode_ms']:>12.1f} {case['separate_decode_ms']:>10.1f} "
                    f"{speedup:>7.1f}x | {max_dp:>8.4f} {max_dlp:>9.4f} "
                    f"{'yes' if agree else 'NO':>6}"
                )
                if args.details:
                    show_worst(
                        shared[0],
                        separate[0],
                        case["shared_request_ms"],
                        case["shared_context_ms"],
                    )
            else:
                saving = s0["prompt_tokens"] / s0["evaluated_tokens"]
                print(
                    f"{state_tokens:>6} {q:>3} | "
                    f"{s0['evaluated_tokens']:>6}/{'(skipped)':<8} {saving:>6.1f}x | "
                    f"{case['shared_decode_ms']:>12.1f} {'-':>10} {'-':>8} | "
                    f"{'-':>8} {'-':>9} {'-':>6}"
                )
            if args.order_check and q > 1:
                # Same questions, same shared prefix, reversed: every question
                # reads the same tokens, only where its cells sit in the batch
                # changes. What moves here moves because of the layout alone.
                reversed_questions = dict(reversed(list(questions.items())))
                try:
                    rev = run_case(
                        url,
                        args.model,
                        state,
                        reversed_questions,
                        "shared_prefix",
                        1,
                        args.timeout,
                    )[0]
                except RequestFailed as e:
                    print(f"{'':>11}reversed order failed: {e}", file=sys.stderr)
                    failed = True
                else:
                    rdp, rdlp, ragree = compare(shared[0], rev)
                    case.update(
                        {
                            "order_max_probability_difference": rdp,
                            "order_max_logprob_difference": rdlp,
                            "order_argmax_agrees": ragree,
                        }
                    )
                    print(
                        f"{'':>11}reversed order: max dP {rdp:.4f}, max dlogP {rdlp:.4f}, "
                        f"argmax {'yes' if ragree else 'NO'}"
                    )
            results.append(case)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "url": args.url,
                    "model": warm["model"],
                    "flash_attn": flash_attn,
                    "repeat": args.repeat,
                    "cases": results,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
