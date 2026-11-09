"""Part C -- support-ticket triage: input guardrails, PII redaction, retries, output validation,
zero-shot vs few-shot comparison, token budget and segment-aware routing."""
import json
import random
import re
import time

import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATS = ["billing", "delivery", "technical", "fraud"]
PII = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
       ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
       ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]
HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

ZERO_SHOT = """You are a support-ticket triage model. CLASSIFY the customer ticket into exactly one category:
billing, delivery, technical, fraud.
Reply with JSON only, no other text: {"category": "<one of the four>", "reason": "<short reason>"}"""

FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: Someone used my card for an order I never placed. -> {"category": "fraud", "reason": "unauthorized use"}
Ticket: The app shows an error when I pay. -> {"category": "technical", "reason": "app error"}
Ticket: I was billed twice for one order. -> {"category": "billing", "reason": "double charge"}"""

jitter = random.Random(42)


def redact(text):
    counts = {}
    for name, pat, ph in PII:
        text, n = pat.subn(ph, text)
        counts[name] = n
    return text, counts


def call_with_retry(messages, ticket_id, log):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages), n
        except TransientLLMError as e:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * jitter.uniform(0.9, 1.15)
            log.append(f"retry ticket={ticket_id} retry_no={n} wait={wait:.4f}s ({e})")
            print(f"  [retry] ticket={ticket_id} retry_no={n} wait={wait:.4f}s")
            time.sleep(wait)


def validate(text):
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (text or "").lower()))
        hits = [c for c in CATS if c in words]
        return ("classified_keyword", hits[0]) if len(hits) == 1 else ("flagged_invalid_output", "")
    if isinstance(data, dict) and data.get("category") in CATS:
        return "classified_json", data["category"]
    return "flagged_invalid_output", ""


def run(tickets, system_prompt, token_budget=None, label=""):
    mock_llm.reset()
    total, rows, retry_log = 0, [], []
    for t in tickets.itertuples():
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": "", "category": "", "tokens_used": 0}
        text = t.text
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            clean, _ = redact(text)
            assert not any(p.search(clean) for _, p, _ in PII), "PII leaked into prompt"
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            resp, _ = call_with_retry(msgs, t.ticket_id, retry_log)
            used = resp.usage.total_tokens
            total += used
            row["tokens_used"] = used
            row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"[{label}] retries={len(retry_log)} total_tokens={total} status counts={df['status'].value_counts().to_dict()}")
    return df, total, retry_log


def summarise(df, total, name):
    cl = df[df["status"].isin(["classified_json", "classified_keyword"])]
    acc = (cl["category"] == cl["true_category"]).mean()
    print(f"{name}: classified {len(cl)} | accuracy {acc:.1%} ({(cl['category'] == cl['true_category']).sum()}/{len(cl)})"
          f" | total tokens {total} | tokens per classified ticket {total / len(cl):.1f}")
    return cl


def route(df, seg):
    prem = seg.set_index("customer_id")["is_premium"].to_dict()
    names = seg.set_index("customer_id")["segment_name"].to_dict()
    out = df.copy()
    out["segment_name"] = out["customer_id"].map(names).fillna("")
    is_prem = out["customer_id"].map(prem).fillna(False).astype(bool)
    out["route"], out["priority"] = "support_queue", "NORMAL"
    out.loc[is_prem, ["route", "priority"]] = ["premium_desk", "HIGH"]
    out.loc[out["category"] == "fraud", ["route", "priority"]] = ["security_team", "HIGH"]
    out.loc[out["status"].isin(HUMAN), ["route", "priority"]] = ["human_review", "NORMAL"]
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]


def main():
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    print(f"tickets: {len(tickets)}")

    # guardrail + redaction report over all 164 tickets
    empty = tickets["text"].str.strip() == ""
    long_ = tickets["text"].map(est_tokens) > MAX_INPUT_TOKENS
    print(f"empty tickets {empty.sum()}; long tickets {long_.sum()} "
          f"(est. tokens {sorted(set(tickets.loc[long_, 'text'].map(est_tokens)))})")
    totals = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
    redacted = []
    for txt in tickets["text"]:
        r, c = redact(txt)
        redacted.append(r)
        for k in totals:
            totals[k] += c[k]
    left = sum(len(p.findall(r)) for r in redacted for _, p, _ in PII)
    assert left == 0
    print(f"redactions: {totals} total {sum(totals.values())}; matches left after redaction: {left}")
    print(f"prompt sizes: zero-shot {est_tokens(ZERO_SHOT)} tokens, few-shot {est_tokens(FEW_SHOT)} tokens")
    assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

    zs, zs_tok, zs_log = run(tickets, ZERO_SHOT, label="zero-shot")
    fs, fs_tok, _ = run(tickets, FEW_SHOT, label="few-shot")
    assert len(zs_log) == 19
    print("\nretry log (zero-shot run):")
    print("\n".join(zs_log))
    print()
    zs_cl = summarise(zs, zs_tok, "zero-shot")
    summarise(fs, fs_tok, "few-shot")
    print("\nzero-shot confusion table (rows true, cols predicted):")
    print(pd.crosstab(zs_cl["true_category"], zs_cl["category"]).to_string())

    seg = pd.read_csv("outputs/customer_segments.csv")
    budget = 6000
    demo, demo_tok, _ = run(tickets, ZERO_SHOT, token_budget=budget, label=f"budget demo {budget}")
    n_exh = int((demo["status"] == "budget_exhausted").sum())
    assert n_exh >= 1
    print(f"budget demo: TOKEN_BUDGET={budget}, tokens spent {demo_tok}, budget_exhausted tickets {n_exh}")
    route(demo, seg).to_csv("outputs/triage_budget_demo.csv", index=False)

    final, final_tok, _ = run(tickets, FEW_SHOT, token_budget=100000, label="final few-shot")
    res = route(final, seg)
    res.to_csv("outputs/triage_results.csv", index=False)
    print(f"\ntriage_results.csv rows {len(res)}; tokens {final_tok}")
    print(res.groupby(["route", "priority"]).size().to_string())


if __name__ == "__main__":
    main()
