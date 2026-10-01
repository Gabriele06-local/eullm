#!/usr/bin/env python3
"""Support triage in LangGraph, with Reflex deciding where each ticket goes.

One request to EuLLM's `POST /v1/systemone` asks three questions about a
ticket, and the ticket is read once for all of them:

  * which team handles it (`choice`, among the teams you describe);
  * is the customer blocked, or facing a deadline today (`noul`);
  * must a person handle it: a legal threat, or someone else in the
    account (`noul`).

The answers are probabilities, and they travel in the graph's state. The
conditional edge after the decision, `route()`, is plain code reading them:
a ticket a person must handle goes to a person, and so does one whose team
the model is unsure of; any other goes to its team's node, where a chat
model on EuLLM drafts the reply. The person's node pauses the graph with
`interrupt()` and shows what the model saw; a person resumes it with a
team, whose node then drafts the reply, or keeps the ticket.

    eullm pull hf.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF:Q4_K_M
    eullm pull qwen3-8b
    eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m
    python examples/decision-langgraph/triage_graph.py --chat-model qwen3-8b
    python examples/decision-langgraph/triage_graph.py --chat-model qwen3-8b --ask

`--teams` takes a JSON object of team name → what the team handles, and
`--tickets` a JSON list of {"from", "subject", "message"}.
"""

import argparse
import json
import os
import sys
import textwrap
import time
from typing import TypedDict

import openai
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from eullm_client import EuLLM, EuLLMError

TEAMS = {
    "billing": "invoices, charges, refunds, payment methods, changes of plan",
    "technical": "bugs, errors, crashes, notes not syncing, the app or the API not working",
    "account": (
        "signing in, passwords, two-factor authentication, lost phones, "
        "changing or closing an account"
    ),
    "sales": "quotes, plans for teams and companies, discounts, partnerships",
}

# Phrased after trying them on the sample tickets: see the README.
BLOCKED = "The customer is blocked or has a deadline today."
PERSON = (
    "A person must handle this ticket: it threatens legal action, or says that "
    "someone else may have got into the account or seen its notes."
)

COMPANY = "Acme Notes, a notes app"

SAMPLES = [
    {
        "from": "Marco Bianchi <marco@example.com>",
        "subject": "Charged twice for my yearly plan",
        "message": "Hello, my card statement shows two payments of 40 euros on 3 "
        "September for the yearly Personal plan. Could you refund the second one? "
        "Thanks, Marco",
    },
    {
        "from": "Giulia Rossi <giulia@example.com>",
        "subject": "Notes not syncing since the update",
        "message": "Since yesterday's update, the notes I write on my phone do not "
        "show up on my laptop. I already signed out and in again on both.",
    },
    {
        "from": "Paolo Greco <paolo@example.com>",
        "subject": "Locked out before a presentation",
        "message": "I lost my phone with the authenticator app and I cannot sign in. "
        "I have a client presentation at 3 pm today and all my notes are in Acme "
        "Notes. Please help!",
    },
    {
        "from": "Studio Conti <office@studioconti.example>",
        "subject": "Quote for 40 seats",
        "message": "We are a law firm of 40 people and would like a quote for the "
        "Team plan, invoiced to our company, with payment by bank transfer.",
    },
    {
        "from": "Anna Esposito <anna@example.com>",
        "subject": "Formal notice",
        "message": "I have asked three times for my data to be deleted under Article "
        "17 of the GDPR. If I do not receive written confirmation within seven days, "
        "my lawyer will file a complaint with the data protection authority.",
    },
    {
        "from": "Luca Ferri <luca@example.com>",
        "subject": "Someone else is reading my notes",
        "message": "I got an email saying my notebook 'Clients' was shared with an "
        "address I do not know, and I see a session from another country that is not "
        "me. What do I do?",
    },
    {
        "from": "Sara Marino <sara@example.com>",
        "subject": "API returns 500",
        "message": "Since 8:00 this morning every call our integration makes to your "
        "API returns error 500. Our whole team is blocked.",
    },
    {
        "from": "Elena Colombo <elena@example.com>",
        "subject": "Change of email",
        "message": "My company changed its name and my email address with it. Can the "
        "invoices and my sign-in use the new address from now on?",
    },
]


class Triage(TypedDict, total=False):
    """The graph's state: the ticket, and what each node added to it."""

    ticket: dict  # from, subject, message
    decision: dict  # what the decision model answered, as probabilities
    priority: str  # "high" or "normal"
    routed_by: str  # "model", or "person" when a person picked the team
    reason: str  # why the ticket went to a person
    team: str  # the team whose node drafted the reply, or "person"
    reply: str  # the draft the chat model wrote


def questions(teams):
    """The three questions of one decision, in `/v1/systemone`'s shape."""
    return {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": teams,
        },
        "blocked": {"type": "noul", "instructions": BLOCKED},
        "person": {"type": "noul", "instructions": PERSON},
    }


def policy(decision, person_threshold, min_team):
    """Where a ticket goes, and why: plain code over the probabilities, so
    it can be read and changed. The thresholds are the flags of this
    script."""
    if decision["person"] >= person_threshold:
        return "person", f"a person must handle it ({decision['person']:.2f})"
    teams = decision["team"]
    best = max(teams, key=teams.get)
    if teams[best] < min_team:
        ranked = sorted(teams.items(), key=lambda kv: -kv[1])[:3]
        return "person", "unsure of the team: " + ", ".join(f"{t} {p:.2f}" for t, p in ranked)
    return best, ""


def build_graph(
    eullm,
    chat,
    teams=TEAMS,
    decision_model=None,
    person_threshold=0.5,
    min_team=0.6,
    blocked_threshold=0.5,
    checkpointer=None,
):
    """decide → a team's node, which drafts the reply, or a person."""
    reserved = {"decide", "person", START, END} & set(teams)
    if reserved:
        raise ValueError(f"a team cannot be called {', '.join(sorted(reserved))}")

    def decide(state):
        ticket = state["ticket"]
        started = time.perf_counter()
        response = eullm.decide(
            {
                "from": ticket.get("from", ""),
                "subject": ticket.get("subject", ""),
                "message": ticket.get("message", ""),
            },
            questions(teams),
            decision_model,
        )
        answers = response["answers"]
        decision = {
            "team": answers["team"]["probabilities"],
            "blocked": answers["blocked"]["noul"],
            "person": answers["person"]["noul"],
            "ms": (time.perf_counter() - started) * 1000,
        }
        return {
            "decision": decision,
            "priority": "high" if decision["blocked"] >= blocked_threshold else "normal",
            "routed_by": "model",
        }

    def route(state):
        return policy(state["decision"], person_threshold, min_team)[0]

    def person(state):
        decision = state["decision"]
        _, reason = policy(decision, person_threshold, min_team)
        # The graph stops here, its state kept by the checkpointer, until a
        # person resumes it with `Command(resume=...)`.
        choice = interrupt(
            {
                "subject": state["ticket"].get("subject", ""),
                "reason": reason,
                "team": decision["team"],
                "blocked": decision["blocked"],
                "person": decision["person"],
            }
        )
        if choice in teams:
            return Command(goto=choice, update={"routed_by": "person", "reason": reason})
        return Command(goto=END, update={"team": "person", "reason": reason})

    def drafter(team):
        def draft(state):
            ticket = state["ticket"]
            system = (
                f"You write replies for the {team} team of {COMPANY}. The team "
                f"handles {teams[team]}. Write a short, friendly reply to the "
                "customer: say what you understood and what happens next. Do not "
                "promise refunds, dates or fixes; the team decides those. End with "
                f"the signature 'The {team} team, Acme Notes'."
            )
            if state["priority"] == "high":
                system += (
                    " The customer is blocked or has a deadline today: say that the "
                    "ticket has priority."
                )
            message = chat.invoke(
                [
                    ("system", system),
                    (
                        "human",
                        f"From: {ticket.get('from', '')}\n"
                        f"Subject: {ticket.get('subject', '')}\n\n"
                        f"{ticket.get('message', '')}",
                    ),
                ]
            )
            return {"team": team, "reply": message.content}

        return draft

    graph = StateGraph(Triage)
    graph.add_node("decide", decide)
    graph.add_node("person", person, destinations=(*teams, END))
    for team in teams:
        graph.add_node(team, drafter(team))
        graph.add_edge(team, END)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", route, [*teams, "person"])
    # A ticket waiting for a person is kept in memory here; an application
    # keeps it in a database instead (langgraph-checkpoint-sqlite or
    # -postgres), so that it survives a restart.
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def load_tickets(path):
    """The tickets of a JSON file, or the samples without one."""
    if not path:
        return SAMPLES
    with open(path, encoding="utf-8") as f:
        tickets = json.load(f)
    if not isinstance(tickets, list) or not all(isinstance(t, dict) for t in tickets):
        raise SystemExit("--tickets must be a JSON list of {from, subject, message}")
    return tickets


def headline(n, ticket, state):
    """One line for a ticket: where it went, and the probabilities why."""
    decision = state["decision"]
    subject = ticket.get("subject", "")
    subject = subject if len(subject) <= 34 else subject[:33] + "…"
    teams = decision["team"]
    best = max(teams, key=teams.get)
    return (
        f"{n:>3}  {subject:<34}  → {state.get('team', 'person'):<10} "
        f"{state['priority']:<7} ({best} {teams[best]:.2f}, "
        f"blocked {decision['blocked']:.2f}, person {decision['person']:.2f})  "
        f"{decision['ms']:.0f} ms"
    )


def ask(teams):
    """A team's name from the person at the terminal, or "" to keep the
    ticket."""
    while True:
        try:
            choice = input(
                f"     which team takes it? ({', '.join(teams)}; Enter keeps it with a person) "
            ).strip()
        except EOFError:
            return ""
        if not choice or choice in teams:
            return choice
        print(f"     no team is called {choice!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:11434", help="EuLLM server URL")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EULLM_API_KEY"),
        help="API key, when the server requires one (default: $EULLM_API_KEY)",
    )
    parser.add_argument(
        "--decision-model", default=None, help="decision model (default: the one loaded)"
    )
    parser.add_argument(
        "--chat-model", required=True, help="the chat model that drafts replies, e.g. qwen3-8b"
    )
    parser.add_argument("--teams", help="JSON file: team name -> what it handles")
    parser.add_argument("--tickets", help="JSON file: a list of {from, subject, message}")
    parser.add_argument(
        "--person-threshold",
        type=float,
        default=0.5,
        help="probability from which a person must handle a ticket (default 0.5)",
    )
    parser.add_argument(
        "--min-team",
        type=float,
        default=0.6,
        help="below this probability for the likeliest team, a person decides (default 0.6)",
    )
    parser.add_argument(
        "--blocked-threshold",
        type=float,
        default=0.5,
        help="probability from which a ticket gets high priority (default 0.5)",
    )
    parser.add_argument(
        "--ask",
        action="store_true",
        help="ask at the terminal which team takes a ticket that went to a person",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout, seconds")
    args = parser.parse_args(argv)

    teams = TEAMS
    if args.teams:
        with open(args.teams, encoding="utf-8") as f:
            teams = json.load(f)
        if not isinstance(teams, dict) or len(teams) < 2:
            parser.error("--teams must be a JSON object with at least two teams")
    tickets = load_tickets(args.tickets)

    eullm = EuLLM(args.url, args.api_key, args.timeout)
    try:
        app = build_graph(
            eullm,
            eullm.chat_model(args.chat_model),
            teams,
            args.decision_model,
            args.person_threshold,
            args.min_team,
            args.blocked_threshold,
        )
    except ValueError as e:
        parser.error(str(e))

    started = time.perf_counter()
    for n, ticket in enumerate(tickets, 1):
        # One thread per ticket: the checkpointer keeps each one's state apart.
        config = {"configurable": {"thread_id": f"ticket-{n}"}}
        try:
            state = app.invoke({"ticket": ticket}, config)
            print(headline(n, ticket, state), flush=True)
            waiting = state.get("__interrupt__")
            if waiting:
                reason = waiting[0].value["reason"]
                if not args.ask:
                    print(f"     waiting for a person: {reason}", flush=True)
                    continue
                print(f"     {reason}")
                state = app.invoke(Command(resume=ask(teams)), config)
                if state.get("reply"):
                    print(f"     a person chose {state['team']}")
                else:
                    print("     kept by a person")
        except EuLLMError as e:
            raise SystemExit(str(e)) from None
        except openai.APIError as e:
            raise SystemExit(f"{args.url} (chat model {args.chat_model}): {e}") from None
        if state.get("reply"):
            print(textwrap.indent(state["reply"].strip(), "     │ ", lambda _: True), flush=True)
    print(f"{len(tickets)} tickets in {time.perf_counter() - started:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
