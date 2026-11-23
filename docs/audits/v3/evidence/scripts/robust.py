import json, sys, traceback
from pathlib import Path
from decision_intelligence.config import Workspace, Environment, load_components, make_retriever, make_checker, make_generator
from decision_intelligence.pipeline import Pipeline
from decision_intelligence.models import ABSTAIN_TEXT
repo = Path(sys.argv[1]); ws = Workspace(root=repo); comp = load_components(ws, Environment.from_env())
# 1. invariants over every committed v3 row (dev + holdout)
bad = []; n = 0
for run in ("dev-r4", "holdout"):
    for l in open(repo / f"experiments/v3/results/{run}/rows.jsonl"):
        r = json.loads(l); n += 1; a = r["answer"]; adm = {d["doc_id"] for d in r["retrieved"] if d["admitted"]}
        cited = {s for c in a["claims"] for s in c["source_ids"]}
        if a["status"] == "ANSWER" and (not a["claims"] or not cited <= adm or not all(c["quote_verified"] for c in a["claims"])): bad.append((run, r["config"], r["question_id"], "answer"))
        if a["status"] == "ABSTAIN" and (a["claims"] or a["text"] != ABSTAIN_TEXT): bad.append((run, r["config"], r["question_id"], "abstain"))
        if a["status"] not in ("ANSWER", "ABSTAIN", "ERROR"): bad.append((run, r["config"], r["question_id"], "status"))
print(f"invariants checked on {n} rows; violations: {bad[:5]} ({len(bad)})")
# 2. malformed / edge inputs through C and F
P = {"C": Pipeline(make_retriever(comp, "rules-v3"), make_generator("extractive-v1", None, comp.protocol), comp.documents),
     "F": Pipeline(make_retriever(comp, "rules-v3"), make_generator("extractive-v2", None, comp.protocol), comp.documents, checker=make_checker(comp))}
inputs = ["", "   ", "?", "a" * 20000, "Quel budget a été approuvé pour le système de planification des patients ?", "What budget was approved?? \x00‮",
          "'; DROP TABLE decisions; --", "What budget was approved for the decision to trial a patient scheduling system? " * 50,
          "Which decisions about the patient scheduling system do not not involve an external supplier?",
          "What budget was approved for the patient scheduling system?"]
for q in inputs:
    for k, p in P.items():
        try:
            res = p.run(q); a = res.answer
            print(f"{k} {a.status.value:7s} claims={len(a.claims)} cited={a.cited_ids()[:4]} | {q[:60]!r}")
        except Exception as e:
            print(f"{k} EXCEPTION {type(e).__name__}: {e} | {q[:60]!r}")
