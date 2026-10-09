#!/usr/bin/env python3
"""Email triage by a decision model through EuLLM's `POST /v1/systemone`.

Every email becomes the state of one request that asks five questions at
once — the state is read once for all of them:

  * which team handles it (`choice`, the teams you describe);
  * what kind of email it is: legitimate, phishing or spam (`choice`);
  * does it need an answer today (`noul`);
  * how upset is the sender (`score`: calm, dissatisfied, angry);
  * does it talk about someone's health (`noul`).

What code can check exactly, code checks: an IBAN, a payment card number or
an Italian tax code in the text is found by pattern and checksum, not asked
about. The model answers with probabilities, and the policy acting on them
is plain code too, so it can be read and changed: phishing and spam are set
aside, an email whose team the model is unsure of goes to a person, urgency
or anger raises the priority, personal data is flagged. Nothing leaves the
server, and every decision is in its audit trail (the email itself only as
a SHA-256).

How a question is put matters. On the 13 sample emails with the Jev-Style
0.8B, "is this phishing?" as a yes/no question put the angry customer, the
lawyer's letter and the job application over 0.7; asked as a choice between
legitimate, phishing and spam, every email came out on the right side. A
yes/no question about "sensitive data" missed both the IBAN and the hospital
stay; the IBAN is now a pattern, and the question is only about health.

    eullm pull hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M
    eullm serve --decision-model jev-style-0.8b-decision-v3-gguf-q4_k_m
    python examples/decision-triage/triage.py              # built-in samples
    python examples/decision-triage/triage.py --dir inbox/ --csv triage.csv
    python examples/decision-triage/triage.py --mbox archive.mbox --teams teams.json

`--teams` takes a JSON object of team name → what the team handles, to
describe your own organisation instead of the sample one. Only the Python
standard library is needed.
"""

import argparse
import csv
import email
import email.policy
import html
import json
import mailbox
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

TEAMS = {
    "amministrazione": "fatture, pagamenti, addebiti, rimborsi",
    "logistica": "spedizioni, consegne, ritardi, pacchi smarriti",
    "resi": "resi, cambi, prodotti difettosi o sbagliati",
    "assistenza_tecnica": "problemi con il sito, l'app, l'account, errori",
    "privacy": "richieste sui dati personali: accesso, rettifica, cancellazione (GDPR)",
    "legale": "diffide, contestazioni formali, avvocati, tribunali",
    "commerciale": "preventivi, offerte, collaborazioni, rivenditori",
    "risorse_umane": "candidature e curriculum",
}

QUESTIONS = {
    "urgente": {
        "type": "noul",
        "instructions": "Il mittente ha bisogno di una risposta entro oggi.",
    },
    "tono": {
        "type": "score",
        "instructions": "Quanto è arrabbiato il mittente?",
        "criteria": ["calmo", "insoddisfatto", "arrabbiato"],
    },
    "salute": {
        "type": "noul",
        "instructions": (
            "L'email parla della salute di una persona: malattie, ricoveri, "
            "interventi, terapie."
        ),
    },
    "tipo": {
        "type": "choice",
        "instructions": "Che tipo di email è?",
        "criteria": {
            "legittima": (
                "un cliente, un fornitore, un avvocato, un candidato o un partner "
                "che scrive davvero all'azienda"
            ),
            "phishing": (
                "si finge una banca, un corriere o un servizio per rubare "
                "credenziali o soldi, con un link o una richiesta di pagamento"
            ),
            "spam": "pubblicità non richiesta",
        },
    },
}

SAMPLES = [
    {
        "from": "Marco Bianchi <marco.bianchi@example.it>",
        "subject": "Addebito doppio sulla fattura di settembre",
        "body": "Buongiorno, sull'estratto conto vedo due addebiti da 89,90 euro per "
        "la stessa fattura n. 2026/0917. Potete verificare e rimborsare quello in "
        "più? Grazie.",
    },
    {
        "from": "Giulia Rossi <giulia.r@example.com>",
        "subject": "Ordine 48213 non ancora arrivato",
        "body": "Salve, il mio ordine doveva arrivare martedì ma il tracciamento è "
        "fermo da cinque giorni. Mi serve per sabato, è un regalo. Sapete dirmi "
        "dov'è il pacco?",
    },
    {
        "from": "Luca Ferri <l.ferri@example.org>",
        "subject": "VERGOGNA",
        "body": "È la terza volta che vi scrivo e nessuno risponde. Le scarpe sono "
        "arrivate rotte, ho pagato 140 euro e voglio i soldi indietro SUBITO. Se "
        "non ricevo risposta entro oggi vi segnalo all'associazione consumatori.",
    },
    {
        "from": "Anna Esposito <anna.esposito@example.it>",
        "subject": "Cancellazione dei miei dati",
        "body": "Ai sensi dell'articolo 17 del GDPR vi chiedo di cancellare tutti i "
        "dati personali che avete su di me e di confermarmelo per iscritto.",
    },
    {
        "from": "Servizio Clienti <sicurezza@banca-verifica-account.com>",
        "subject": "Il tuo conto è stato sospeso",
        "body": "Gentile cliente, abbiamo rilevato un accesso anomalo. Per evitare "
        "il blocco definitivo verifica subito le tue credenziali al link "
        "http://verifica-conto-sicuro.example.net entro 24 ore.",
    },
    {
        "from": "Studio Legale Conti <avv.conti@example.it>",
        "subject": "Diffida ad adempiere – vostro cliente sig. Romano",
        "body": "In nome e per conto del sig. Romano vi diffido a consegnare la "
        "merce pagata il 3 settembre entro 15 giorni dal ricevimento della "
        "presente, in difetto procederemo nelle sedi opportune.",
    },
    {
        "from": "Paolo Greco <paolo.greco@example.com>",
        "subject": "Non riesco ad accedere",
        "body": "Da ieri l'app mi dice 'errore 500' quando provo a fare il login. "
        "Ho già reinstallato. Sistema operativo Android 15.",
    },
    {
        "from": "Sara Marino <sara.marino@example.it>",
        "subject": "Reso taglia sbagliata",
        "body": "Ciao, ho ricevuto la giacca in taglia M invece di L. Come faccio il "
        "cambio? Per il rimborso eventuale l'IBAN è IT60X0542811101000000123456.",
    },
    {
        "from": "Negozio Alpi Sport <acquisti@alpisport.example>",
        "subject": "Richiesta listino rivenditori",
        "body": "Buongiorno, abbiamo tre negozi in Trentino e vorremmo il vostro "
        "listino per rivenditori e le condizioni per ordini sopra i 5.000 euro.",
    },
    {
        "from": "Fatturazione <billing@corriere-espresso-it.example.net>",
        "subject": "Fattura non pagata – spedizione bloccata",
        "body": "La sua spedizione è bloccata in dogana. Per sbloccarla paghi 2,99 "
        "euro di diritti entro oggi a questo link: "
        "http://pagamento-dogana.example.net/p",
    },
    {
        "from": "Ufficio crediti Tessuti Riva <crediti@tessutiriva.example.it>",
        "subject": "Sollecito fattura 2026/311 scaduta",
        "body": "Buongiorno, la fattura 2026/311 di 1.240 euro è scaduta il 15 "
        "settembre. Vi preghiamo di saldarla con bonifico alle coordinate indicate "
        "in fattura.",
    },
    {
        "from": "Maria Lombardi <maria.lombardi@example.it>",
        "subject": "Ritiro pacco",
        "body": "Buongiorno, sono ricoverata in ospedale dopo un intervento al "
        "ginocchio e non potrò ritirare il pacco fino a lunedì prossimo. Potete "
        "tenerlo in deposito?",
    },
    {
        "from": "Elena Colombo <elena.colombo@example.com>",
        "subject": "Candidatura magazziniere",
        "body": "Buongiorno, allego il mio curriculum per la posizione di "
        "magazziniere pubblicata sul vostro sito. Sono disponibile da subito.",
    },
]


def iban_ok(candidate):
    """ISO 13616 check digits: the IBAN, rotated and read as a number, is 1
    modulo 97."""
    rotated = candidate[4:] + candidate[:4]
    digits = "".join(str(int(c, 36)) for c in rotated)
    return int(digits) % 97 == 1


def luhn_ok(number):
    total = 0
    for i, d in enumerate(reversed(number)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def personal_data(text):
    """The personal data code can find exactly: IBANs and payment card
    numbers whose check digits hold, and Italian tax codes."""
    found = []
    for match in re.finditer(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b", text):
        if iban_ok(match.group().replace(" ", "")):
            found.append("IBAN")
            break
    for match in re.finditer(r"\b\d(?:[ -]?\d){12,18}\b", text):
        if luhn_ok(re.sub(r"[ -]", "", match.group())):
            found.append("numero di carta")
            break
    if re.search(r"\b[A-Z]{6}\d{2}[A-EHLMPR-T]\d{2}[A-Z]\d{3}[A-Z]\b", text.upper()):
        found.append("codice fiscale")
    return found


def text_of(message):
    """The readable text of an email: its plain-text part, else its HTML
    with the tags stripped."""
    part = message.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    content = part.get_content()
    if part.get_content_type() == "text/html":
        content = re.sub(r"(?is)<(script|style).*?</\1>", " ", content)
        content = html.unescape(re.sub(r"<[^>]+>", " ", content))
    return re.sub(r"\s+", " ", content).strip()


def load(args):
    """The emails to triage, as {from, subject, body}."""
    if args.dir:
        paths = sorted(pathlib.Path(args.dir).glob("*.eml"))
        found = []
        for path in paths:
            with open(path, "rb") as f:
                message = email.message_from_binary_file(f, policy=email.policy.default)
            found.append(
                {
                    "from": str(message.get("From", "")),
                    "subject": str(message.get("Subject", "")),
                    "body": text_of(message),
                }
            )
        return found
    if args.mbox:
        # mailbox.mbox creates the file when missing (create=True is the
        # default), so a typo'd path silently grew a stray 0-byte file and
        # the run ended with "no email found" for mail that never existed.
        if not os.path.isfile(args.mbox):
            raise SystemExit(f"no such mbox file: {args.mbox}")
        box = mailbox.mbox(args.mbox, factory=None)
        found = []
        for raw in box:
            message = email.message_from_bytes(
                raw.as_bytes(), policy=email.policy.default
            )
            found.append(
                {
                    "from": str(message.get("From", "")),
                    "subject": str(message.get("Subject", "")),
                    "body": text_of(message),
                }
            )
        return found
    return SAMPLES


def ask(args, teams, mail):
    """One request: the email as the state, the five questions."""
    questions = {
        "reparto": {
            "type": "choice",
            "instructions": "A quale reparto va inoltrata questa email?",
            "criteria": teams,
        },
        **QUESTIONS,
    }
    body = mail["body"][: args.max_chars]
    payload = {
        "state": {"mittente": mail["from"], "oggetto": mail["subject"], "testo": body},
        "questions": questions,
    }
    if args.model:
        payload["model"] = args.model
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/systemone",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            answer = json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(f"HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise SystemExit(
            f"{args.url}: {e.reason} — is `eullm serve` running?"
        ) from None
    answer["_ms"] = (time.perf_counter() - started) * 1000.0
    return answer


def policy(args, mail, answers):
    """What to do with the email, from the model's answers and what code
    found in it. Plain code: the thresholds are the flags of this script."""
    team = answers["reparto"]
    flags = personal_data(mail["subject"] + " " + mail["body"])
    if answers["salute"]["noul"] >= args.threshold:
        flags.append("dati sulla salute")
    kind = answers["tipo"]["probabilities"]
    if kind["phishing"] >= args.quarantine:
        return "quarantena", "", flags
    if kind["spam"] >= args.quarantine:
        return "spam", "", flags
    urgent = answers["urgente"]["noul"] >= args.threshold
    angry = answers["tono"]["score"] >= 1.5
    priority = "alta" if urgent or angry else "normale"
    if team["probabilities"][team["choice"]] < args.min_team:
        return "revisione umana", priority, flags
    return team["choice"], priority, flags


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--url", default="http://localhost:11434", help="EuLLM server URL"
    )
    parser.add_argument(
        "--model", default=None, help="decision model (default: the one loaded)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EULLM_API_KEY"),
        help="API key, when the server requires one (default: $EULLM_API_KEY)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dir", help="a folder of .eml files")
    source.add_argument("--mbox", help="an mbox file")
    parser.add_argument("--teams", help="JSON file: team name -> what it handles")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="probability from which a yes/no answer counts as yes (default 0.5)",
    )
    parser.add_argument(
        "--quarantine",
        type=float,
        default=0.7,
        help="phishing or spam probability from which an email is set aside (default 0.7)",
    )
    parser.add_argument(
        "--min-team",
        type=float,
        default=0.4,
        help="below this probability for the chosen team, a person decides (default 0.4)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="email text sent at most (default 4000)",
    )
    parser.add_argument("--csv", help="write one row per email to this CSV file")
    parser.add_argument(
        "--json", help="write every email's full answers to this JSON file"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="per-request timeout, seconds"
    )
    args = parser.parse_args()

    teams = TEAMS
    if args.teams:
        with open(args.teams, encoding="utf-8") as f:
            teams = json.load(f)
        if not isinstance(teams, dict) or len(teams) < 2:
            parser.error("--teams must be a JSON object with at least two teams")
    mails = load(args)
    if not mails:
        raise SystemExit("no email found")

    rows = []
    full = []
    for n, mail in enumerate(mails, 1):
        response = ask(args, teams, mail)
        answers = response["answers"]
        action, priority, flags = policy(args, mail, answers)
        team = answers["reparto"]
        row = {
            "n": n,
            "oggetto": mail["subject"],
            "azione": action,
            "priorita": priority,
            "reparto_p": round(team["probabilities"][team["choice"]], 2),
            "urgente_p": round(answers["urgente"]["noul"], 2),
            "tono": round(answers["tono"]["score"], 2),
            "phishing_p": round(answers["tipo"]["probabilities"]["phishing"], 2),
            "segnalazioni": "; ".join(flags),
            "ms": round(response["_ms"]),
        }
        rows.append(row)
        full.append({"email": mail, "row": row, "response": response})
        subject = (
            mail["subject"]
            if len(mail["subject"]) <= 38
            else mail["subject"][:37] + "…"
        )
        print(
            f"{n:>3}  {subject:<38}  → {action:<18} {priority:<8} "
            f"(p {row['reparto_p']:.2f}, urgente {row['urgente_p']:.2f}, "
            f"tono {row['tono']:.1f}, phishing {row['phishing_p']:.2f})"
            + (f"  ⚠ {row['segnalazioni']}" if flags else "")
            + f"  {row['ms']} ms",
            flush=True,
        )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(full, f, ensure_ascii=False, indent=2)
    total = sum(r["ms"] for r in rows)
    print(f"{len(rows)} email in {total / 1000:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
