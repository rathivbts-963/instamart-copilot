"""Part D tasks 5-8 -- refund agent: tools, ReAct loop, safety limits, checkpoint/resume."""
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm  # noqa: E402
from d_rag import retrieve  # noqa: E402  (same retrieval as the policy assistant)

DB = "data/instamart.db"
LEDGER = "outputs/refund_ledger.jsonl"
RUNS = "outputs/agent_runs.jsonl"
STATE = "state"
SYSTEM = ("You are the Instamart refund AGENT. Decide refund requests by calling tools: look up the order, "
          "check the refund policy, calculate the refund, then issue it or escalate it to a human. "
          "Finish with one line: REFUND, ESCALATE, DENIED or NOT_FOUND, a colon, and a short reason.")


# ---------------------------------------------------------------- tools
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or float(delivery_minutes) <= 45:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}


def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the Instamart database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve the three most relevant policy passages for a query.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                     "delivery_minutes": {"type": ["number", "null"]}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue a refund of at most INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ---------------------------------------------------------------- loop
class SimulatedCrash(Exception):
    pass


def _save(state):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)


def _fallback(state, status, reason):
    tools = ", ".join(state["tools_executed"]) or "none"
    return status, (f"HANDOFF to human agent: {reason}. Request: {state['messages'][1]['content']!r}. "
                    f"Tools already executed: {tools}.")


def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tool=None):
    t0 = time.monotonic()
    msgs = state["messages"]
    while True:
        if time.monotonic() - t0 >= time_limit_seconds:
            status, final = _fallback(state, "FALLBACK_TIMEOUT", f"time limit of {time_limit_seconds}s reached")
            break
        if state["llm_calls"] >= max_iterations:
            status, final = _fallback(state, "FALLBACK_MAX_ITERATIONS", f"no answer after {max_iterations} model calls")
            break
        resp = mock_llm.chat(msgs, TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit:
            status, final = _fallback(state, "FALLBACK_TOKEN_LIMIT", f"token limit {token_limit} exceeded")
            break
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content
            status = final.split(":", 1)[0].strip()
            break
        for tc in msg.tool_calls:
            msgs.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
            obs = TOOL_MAP[tc.name](**tc.arguments)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
            state["step"] += 1
            state["tools_executed"].append(tc.name)
            state["steps"].append({"step": state["step"], "tool": tc.name, "arguments": tc.arguments,
                                   "observation": obs})
            _save(state)
            if crash_after_tool is not None and state["step"] == crash_after_tool:
                raise SimulatedCrash(f"simulated crash after tool #{state['step']} ({tc.name})")
    refund = next((s["observation"]["refund_inr"] for s in state["steps"]
                   if s["tool"] in ("issue_refund", "escalate_to_human")), 0.0)
    state.update(status=status, final_answer=final, refund_inr=refund)
    _save(state)
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": final, "refund_inr": refund, "steps": state["steps"],
            "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"],
            "tools_executed": state["tools_executed"]}


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tool=None):
    state = {"run_id": run_id or scenario_id, "scenario_id": scenario_id,
             "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                        "time_limit_seconds": time_limit_seconds},
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": []}
    return _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tool)


def resume_agent(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    lim = state["limits"]
    return _loop(state, lim["max_iterations"], lim["token_limit"], lim["time_limit_seconds"])


def log_run(res, note=""):
    rec = {k: res[k] for k in ("run_id", "scenario_id", "status", "final_answer", "refund_inr",
                               "llm_calls", "tokens_used", "steps")}
    if note:
        rec["note"] = note
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    os.makedirs("outputs", exist_ok=True)
    for p in (LEDGER, RUNS):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(STATE, ignore_errors=True)

    sc = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
    rows = []
    for s in sc.itertuples():
        res = run_agent(s.message, s.scenario_id)
        log_run(res)
        exp_status = "FALLBACK_MAX_ITERATIONS" if s.expected_outcome == "FALLBACK" else s.expected_outcome
        assert res["status"] == exp_status, (s.scenario_id, res["status"])
        assert abs(res["refund_inr"] - float(s.expected_refund_inr)) < 0.01
        rows.append({"scenario": s.scenario_id, "expected": s.expected_outcome, "status": res["status"],
                     "refund_inr": f"{res['refund_inr']:.2f}", "llm_calls": res["llm_calls"],
                     "tokens_used": res["tokens_used"], "tools": " > ".join(res["tools_executed"])})
    print(pd.DataFrame(rows).to_string(index=False))

    s1 = sc.iloc[0]
    tok = run_agent(s1.message, "S1", token_limit=200, run_id="S1-token-limit-demo")
    log_run(tok, "demo: token_limit=200")
    print(f"\nS1 with token_limit=200 -> {tok['status']} after {tok['llm_calls']} calls, "
          f"{tok['tokens_used']} tokens, tools {tok['tools_executed']}")
    tmo = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
    log_run(tmo, "demo: time_limit_seconds=0")
    print(f"S1 with time_limit_seconds=0 -> {tmo['status']} after {tmo['llm_calls']} model calls")
    print(f"  handoff summary: {tmo['final_answer']}")
    assert tok["status"] == "FALLBACK_TOKEN_LIMIT" and tmo["status"] == "FALLBACK_TIMEOUT"

    run_id = "S1-crash-resume-demo"
    try:
        run_agent(s1.message, "S1", run_id=run_id, crash_after_tool=2)
    except SimulatedCrash as e:
        with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
            before = json.load(f)["tools_executed"]
        print(f"\ncrash: {e}; tools executed before crash: {before}")
    res = resume_agent(run_id)
    log_run(res, "demo: crash after 2nd tool, then resume_agent")
    after = res["tools_executed"][len(before):]
    print(f"resumed: tools executed after resume: {after}")
    print(f"all tools across crash+resume: {res['tools_executed']} -> {res['status']} INR {res['refund_inr']:.2f}")
    assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(res["refund_inr"] - 128.25) < 0.01
    print("\nALL AGENT CHECKS PASSED")


if __name__ == "__main__":
    main()
