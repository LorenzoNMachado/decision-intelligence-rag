import sys
from pathlib import Path
from decision_intelligence.config import Workspace, Environment, load_components, make_retriever, make_generator
from decision_intelligence.sufficiency import EvidenceSufficiencyChecker
from decision_intelligence.pipeline import Pipeline
ws = Workspace(root=Path(sys.argv[1])); comp = load_components(ws, Environment.from_env())
Q = [q for q in ws.load_questions() if q.split == "holdout" and q.answerable]
for label, scopes in (("with subject/domain fields (frozen)", comp.scopes), ("without them (scopes=())", ())):
    p = Pipeline(make_retriever(comp, "rules-v3"), make_generator("extractive-v2", None, comp.protocol), comp.documents,
                 checker=EvidenceSufficiencyChecker(entities=comp.entities, scopes=scopes))
    prec = []; correct = 0; abst = 0
    for q in Q:
        a = p.run(q.text).answer
        if a.status.value == "ABSTAIN": abst += 1; continue
        cited = a.cited_ids(); prec.append(sum(c in q.expected_doc_ids for c in cited) / len(cited))
        t = a.text.lower(); correct += all(k.lower() in t for k in q.key_facts) and not any(f.lower() in t for f in q.forbidden_facts)
    print(f"F {label}: precision {sum(prec)/len(prec):.3f}, strict correct {correct}/100, false abstention {abst}/100")
