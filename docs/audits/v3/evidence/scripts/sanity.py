import json, sys
from pathlib import Path
from decision_intelligence.config import Workspace, Environment, load_components, make_retriever, make_checker, make_generator
from decision_intelligence.pipeline import Pipeline
repo = Path(sys.argv[1]); ws = Workspace(root=repo); comp = load_components(ws, Environment.from_env())
rows = [json.loads(l) for l in open(repo/"experiments/v3/results/dev-r4/rows.jsonl") if json.loads(l)["config"] == "F"]
qs = {q.id: q for q in ws.load_questions()}
p = Pipeline(make_retriever(comp, "rules-v3"), make_generator("extractive-v2", None, comp.protocol), comp.documents, checker=make_checker(comp))
same = 0
for r in rows:
    res = p.run(qs[r["question_id"]].text)
    same += (res.answer.status.value == r["answer"]["status"] and res.answer.text == r["answer"]["text"])
print("dev-r4 F rows reproduced through the public API:", same, "/", len(rows))
