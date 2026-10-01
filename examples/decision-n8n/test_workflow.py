"""The n8n workflow, checked without n8n: the file parses, the nodes are the
ones the README describes, in the versions it names, wired as it says; the
request body is JSON once n8n fills in the ticket, and asks what the
LangGraph triage asks; the Switch's rules send each decision where the
policy says. Standard library only:

    python -m unittest discover -s examples/decision-n8n
"""

import ast
import json
import operator
import pathlib
import re
import unittest

HERE = pathlib.Path(__file__).resolve().parent
WORKFLOW = HERE / "support-triage.json"
TEAMS = ["billing", "technical", "account", "sales"]
BRANCHES = {"person": "To a person", **{team: team.capitalize() for team in TEAMS}}


def lookup(path, item):
    """The value at `$json.a.b`, or at `$json.a.b[<another such path>]`: the
    only expressions the rules use. Anything else fails the test rather than
    being read wrongly."""
    m = re.fullmatch(r"\$json((?:\.\w+)+)(?:\[(.+)\])?", path.strip())
    if not m:
        raise ValueError(f"not an expression this test reads: {path}")
    value = item
    for key in m.group(1).split(".")[1:]:
        value = value[key]
    if m.group(2):
        value = value[lookup(m.group(2), item)]
    return value


OPERATIONS = {
    ("number", "gte"): operator.ge,
    ("number", "lt"): operator.lt,
    ("string", "equals"): operator.eq,
}


def response(team, blocked, person):
    """A /v1/systemone response to the workflow's three questions."""
    choice = max(team, key=team.get)
    return {
        "answers": {
            "team": {"type": "choice", "choice": choice, "probabilities": team},
            "blocked": {"type": "noul", "noul": blocked},
            "person": {"type": "noul", "noul": person},
        }
    }


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        self.nodes = {node["name"]: node for node in self.workflow["nodes"]}

    def targets(self, name):
        outputs = self.workflow["connections"][name]["main"]
        return [[target["node"] for target in output] for output in outputs]

    def route(self, item):
        """The output the Switch sends `item` to: the first rule that
        matches, else the fallback."""
        for rule in self.nodes["Route"]["parameters"]["rules"]["values"]:
            conditions = rule["conditions"]
            results = []
            for condition in conditions["conditions"]:
                left = re.fullmatch(r"=\{\{(.+)\}\}", condition["leftValue"]).group(1)
                kind = (condition["operator"]["type"], condition["operator"]["operation"])
                results.append(OPERATIONS[kind](lookup(left, item), condition["rightValue"]))
            if {"and": all, "or": any}[conditions["combinator"]](results):
                return rule["outputKey"]
        return "unmatched"

    def payload(self):
        """The request body, with the one expression in it filled in."""
        body = self.nodes["Ask Reflex"]["parameters"]["jsonBody"]
        self.assertTrue(body.startswith("="), "an expression in n8n starts with =")
        expressions = re.findall(r"\{\{(.*?)\}\}", body, re.S)
        self.assertEqual(len(expressions), 1)
        ticket = {"from": "a", "subject": 'quotes " and }} braces', "message": "two\nlines"}
        filled = re.sub(r"\{\{.*?\}\}", lambda _: json.dumps(ticket), body[1:], flags=re.S)
        return json.loads(filled)

    def test_the_nodes_in_their_current_versions(self):
        kinds = {name: (node["type"], node["typeVersion"]) for name, node in self.nodes.items()}
        self.assertEqual(kinds["Ticket"], ("n8n-nodes-base.webhook", 2.1))
        self.assertEqual(kinds["Ask Reflex"], ("n8n-nodes-base.httpRequest", 4.5))
        self.assertEqual(kinds["Route"], ("n8n-nodes-base.switch", 3.4))
        for name in BRANCHES.values():
            self.assertEqual(kinds[name], ("n8n-nodes-base.set", 3.5))
        ids = [node["id"] for node in self.workflow["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_webhook_then_reflex_then_switch_then_a_branch(self):
        self.assertEqual(self.targets("Ticket"), [["Ask Reflex"]])
        self.assertEqual(self.targets("Ask Reflex"), [["Route"]])
        rules = self.nodes["Route"]["parameters"]["rules"]["values"]
        self.assertEqual([rule["outputKey"] for rule in rules], ["person", *TEAMS])
        # One output per rule, in order, then the fallback, to a person too.
        self.assertEqual(self.nodes["Route"]["parameters"]["options"]["fallbackOutput"], "extra")
        wired = [*(BRANCHES[rule["outputKey"]] for rule in rules), "To a person"]
        self.assertEqual(self.targets("Route"), [[name] for name in wired])
        for source, outputs in self.workflow["connections"].items():
            self.assertIn(source, self.nodes)
            for output in outputs["main"]:
                for target in output:
                    self.assertIn(target["node"], self.nodes)
                    self.assertEqual((target["type"], target["index"]), ("main", 0))

    def test_the_webhook_answers_with_the_branch_s_fields(self):
        p = self.nodes["Ticket"]["parameters"]
        self.assertEqual((p["httpMethod"], p["path"]), ("POST", "eullm-triage"))
        self.assertEqual((p["responseMode"], p["responseData"]), ("lastNode", "firstEntryJson"))

    def test_the_key_is_a_header_credential_and_never_in_the_file(self):
        node = self.nodes["Ask Reflex"]
        p = node["parameters"]
        self.assertEqual(p["method"], "POST")
        self.assertTrue(p["url"].endswith("/v1/systemone"))
        self.assertEqual(p["authentication"], "genericCredentialType")
        self.assertEqual(p["genericAuthType"], "httpHeaderAuth")
        self.assertNotIn("credentials", node)
        self.assertNotIn("sendHeaders", p)
        # A decision on a CPU takes seconds: more than n8n's 10 s default.
        self.assertGreaterEqual(p["options"]["timeout"], 30000)

    def test_the_body_sends_the_ticket_and_three_questions(self):
        payload = self.payload()
        self.assertEqual(payload["state"]["subject"], 'quotes " and }} braces')
        questions = payload["questions"]
        kinds = {name: q["type"] for name, q in questions.items()}
        self.assertEqual(kinds, {"team": "choice", "blocked": "noul", "person": "noul"})
        self.assertEqual(list(questions["team"]["criteria"]), TEAMS)
        state = re.findall(r"\{\{(.*?)\}\}", self.nodes["Ask Reflex"]["parameters"]["jsonBody"])
        self.assertIn("JSON.stringify", state[0])  # the ticket's text, escaped as JSON

    def test_the_questions_are_the_langgraph_triage_s(self):
        source = HERE.parent / "decision-langgraph" / "triage_graph.py"
        if not source.exists():
            self.skipTest("the LangGraph example is not next to this one")
        constants = {}
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ("TEAMS", "BLOCKED", "PERSON"):
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
        questions = self.payload()["questions"]
        self.assertEqual(questions["team"]["criteria"], constants["TEAMS"])
        self.assertEqual(questions["blocked"]["instructions"], constants["BLOCKED"])
        self.assertEqual(questions["person"]["instructions"], constants["PERSON"])

    def test_the_rules_route_as_the_policy_does(self):
        def teams(billing, technical, account, sales):
            return {"billing": billing, "technical": technical, "account": account, "sales": sales}

        clear = teams(0.9, 0.05, 0.03, 0.02)
        cases = [
            (response(clear, 0.1, 0.1), "billing"),
            (response(teams(0.0, 0.05, 0.94, 0.01), 0.96, 0.12), "account"),
            (response(teams(0.09, 0.0, 0.01, 0.9), 0.0, 0.0), "sales"),
            # A person must handle it, however sure the team.
            (response(clear, 0.1, 0.77), "person"),
            (response(clear, 0.1, 0.5), "person"),
            # Unsure of the team: below 0.6 for the likeliest.
            (response(teams(0.52, 0.01, 0.46, 0.01), 0.0, 0.06), "person"),
            (response(teams(0.6, 0.01, 0.38, 0.01), 0.0, 0.06), "billing"),
        ]
        for item, want in cases:
            self.assertEqual(self.route(item), want, item)

    def fields(self, name):
        assignments = self.nodes[name]["parameters"]["assignments"]["assignments"]
        return {a["name"]: a for a in assignments}

    def test_every_branch_says_where_the_ticket_went(self):
        for route, name in BRANCHES.items():
            fields = self.fields(name)
            self.assertEqual(fields["route"]["value"], route)
            self.assertEqual(
                fields["priority"]["value"],
                "={{ $json.answers.blocked.noul >= 0.5 ? 'high' : 'normal' }}",
            )
            # The ticket itself is the webhook's, reached by its paired item.
            self.assertEqual(
                fields["subject"]["value"], "={{ $('Ticket').item.json.body.subject }}"
            )
            for field in ("team", "blocked", "person"):
                self.assertIn(f"$json.answers.{field}", fields[field]["value"])
            ids = [a["id"] for a in fields.values()]
            self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("reason", self.fields("To a person"))


if __name__ == "__main__":
    unittest.main()
