# decision-n8n — Reflex decides, n8n routes

An [n8n](https://n8n.io) workflow in which a support ticket posted to a
webhook goes to Reflex, EuLLM's decision primitive (`POST /v1/systemone`),
and a Switch node routes it on the probabilities that come back: to a person
when the model says a person must handle it, or is unsure of the team, and
otherwise to the team's branch.

```
Ticket ──→ Ask Reflex ──→ Route ──┬─→ To a person   a person must handle it, or the team is unsure
                                  ├─→ Billing
                                  ├─→ Technical
                                  ├─→ Account
                                  ├─→ Sales
                                  └─→ To a person   anything no rule matches
```

The questions, the teams and the thresholds are those of the LangGraph
triage in [`../decision-langgraph/`](../decision-langgraph/README.md), where
they were tried on sample tickets; a test checks they stay the same.

| Node | What it does |
|---|---|
| **Ticket** | Webhook, `POST /webhook/eullm-triage`. Answers the caller when the last node has run, with that node's fields. |
| **Ask Reflex** | HTTP Request, `POST /v1/systemone`: the ticket's `from`, `subject` and `message` as the state, and three questions about it, read once for all three — which team (`choice`), is the customer blocked or facing a deadline today (`noul`), must a person handle it (`noul`). The API key goes as a header, from an n8n credential: it is never in the workflow file. |
| **Route** | Switch, first matching rule wins. `person`: P(person) ≥ 0.5, or the likeliest team below 0.6; then one rule per team on the model's choice. Anything else leaves by the fallback output, to a person too. |
| **To a person**, **Billing**, **Technical**, **Account**, **Sales** | Edit Fields: where the ticket went, its priority (high when P(blocked) ≥ 0.5), its subject and the probabilities. They stand where your own steps go: a message to the team's channel, a ticket in your help desk, an email. |

## Setup

It needs n8n 2.33 or later: the workflow uses the current versions of its
nodes (HTTP Request 4.5, Switch 3.4, Edit Fields 3.5, Webhook 2.1), and was
made against n8n 2.41.5. On an earlier n8n 2.x, set the HTTP Request node's
`"typeVersion"` to `4.4` and the Edit Fields nodes' to `3.4` in the file
before importing; their parameters are the same.

**EuLLM, with a key for n8n.** n8n usually runs in Docker, and on Linux its
requests then reach EuLLM from the container's network, not from loopback,
the only address EuLLM accepts by default. A key admits a request from any
address, and its id goes into each decision's audit record:

```bash
eullm pull hf.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF:Q4_K_M
secret=$(openssl rand -hex 24); echo "$secret"     # the value for n8n's credential
EULLM_API_KEYS="n8n:$secret" eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m
```

For a server that keeps running, put the key in a file instead
(`EULLM_API_KEYS_FILE`, one `id:secret` per line, `chmod 600`): an
environment variable can be read from `/proc`. See "API keys and quotas" in
[the engine guide](../../docs/engine-guide.md).

**n8n in Docker, EuLLM on the host.** The workflow calls
`http://host.docker.internal:11434/v1/systemone`. Docker Desktop (macOS,
Windows) knows that name; on Linux, start the container with
`--add-host=host.docker.internal:host-gateway`:

```bash
docker run -it --rm --name n8n -p 5678:5678 \
  --add-host=host.docker.internal:host-gateway \
  -v n8n_data:/home/node/.n8n n8nio/n8n
```

With Docker Compose, the same goes under the n8n service as
`extra_hosts: ["host.docker.internal:host-gateway"]`. An n8n installed with
npm on the same machine reaches EuLLM at `http://127.0.0.1:11434`, and needs
no key: change the URL in **Ask Reflex**. Write `127.0.0.1`, not
`localhost`, which Node may try over IPv6 first, where EuLLM is not
listening.

## Import

1. In n8n, create a workflow, open the menu under the three dots at the top
   right, and choose **Import from File**: `examples/decision-n8n/support-triage.json`.
2. Open **Ask Reflex**. Check the URL, and create the Header Auth credential
   it asks for (**Create new credential**): **Name** `X-Api-Key`, **Value**
   the secret. Against a server without keys, any value will do.
3. Open **Ticket**, select **Listen for test event**, and send it a ticket:

   ```bash
   curl -s http://localhost:5678/webhook-test/eullm-triage \
     -H 'Content-Type: application/json' -d '{
       "from": "Paolo Greco <paolo@example.com>",
       "subject": "Locked out before a presentation",
       "message": "I lost my phone with the authenticator app and I cannot sign in. I have a client presentation at 3 pm today and all my notes are in Acme Notes. Please help!"
     }'
   ```

   With the Jev-Style 2B, on a CPU here, this ticket goes to account (0.94)
   with high priority (blocked 0.96); the reply is the **Account** node's
   fields, rounded here:

   ```json
   {"route": "account", "priority": "high", "subject": "Locked out before a presentation",
    "team": {"billing": 0.003, "technical": 0.048, "account": 0.944, "sales": 0.004},
    "blocked": 0.96, "person": 0.12}
   ```

4. **Publish** the workflow. Tickets then go to the production URL,
   `http://localhost:5678/webhook/eullm-triage`, and each run is under
   **Executions**.

A decision on a GPU takes tens of milliseconds; on a CPU, seconds. **Ask
Reflex** waits up to 60 s, under **Options → Timeout**.

## Changing it

- **The thresholds** are numbers in **Route**'s rules (0.5 and 0.6) and in
  each branch's `priority` field (0.5). Set them after trying the questions
  on tickets whose right answer you know.
- **The teams** are named in two places: the `criteria` of the `team`
  question in **Ask Reflex**'s body, and **Route**'s rules, one per team,
  each with a branch of its own. Change both, or a team the model picks has
  no rule and its tickets go to a person by the fallback output.
- **The questions** are plain text in **Ask Reflex**'s body. How a question
  is put matters as much as the model: see what was measured in
  [`../decision-langgraph/`](../decision-langgraph/README.md#support-triage-triage_graphpy).

## Test

```bash
python -m unittest discover -s examples/decision-n8n
```

Standard library only, no n8n: the file parses, the nodes are the ones above
in those versions, wired in that order; the body is valid JSON once the
ticket is filled in and asks what the LangGraph triage asks; the rules,
read the way the Switch reads them, send sample decisions where the policy
says.

The workflow was also checked with n8n's own code, the packages n8n 2.41.5
ships: its workflow validator (`@n8n/workflow-sdk`) with every node's
parameter schema finds nothing to report; its expression engine turns the
body into valid JSON for a ticket with quotes, braces, newlines and emoji in
it; its filter routes the decisions above as the policy does. It was not
imported into a running n8n.
