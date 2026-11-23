import sys
from pathlib import Path
from decision_intelligence.config import Workspace, Environment, load_components, make_retriever, make_checker, make_generator
from decision_intelligence.pipeline import Pipeline
ws = Workspace(root=Path(sys.argv[1])); comp = load_components(ws, Environment.from_env())
p = Pipeline(make_retriever(comp, "rules-v3"), make_generator("extractive-v2", None, comp.protocol), comp.documents, checker=make_checker(comp))
gt = {d: comp.documents[d].ground_truth_signals["third_party_dependency"] for d in [f"DEC-00{i}" for i in range(1,7)]}
print("generator labels third_party for DEC-001..006:", gt)
for q in ["Which decisions about the patient scheduling system involve no external supplier?",
          "Which decisions about the patient scheduling system do not not involve an external supplier?",
          "Which decisions about the patient scheduling system involve an external supplier?"]:
    r = p.run(q); print(r.answer.status.value, r.answer.cited_ids(), [ (s.signal.value, s.polarity.value) for s in r.retrieval.intent.signals], "|", q)
r = p.run("What budget was approved??"); print("no-subject question ->", r.answer.status.value, r.answer.text[:150].replace("\n"," / "))
