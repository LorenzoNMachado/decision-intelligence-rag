import dataclasses, importlib.util, json, sys
from pathlib import Path
repo, cases_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("cases", cases_path); cases = importlib.util.module_from_spec(spec); spec.loader.exec_module(cases)
from decision_intelligence.config import Workspace, Components, make_retriever, make_checker, make_generator, HashingEmbedder
from decision_intelligence.corpus.generator import DOMAINS
from decision_intelligence.corpus.chunking import chunk_corpus
from decision_intelligence.semantic_signals.classifier import RuleBasedSignalClassifier, ScopedSignalClassifier, entity_names
from decision_intelligence.semantic_signals.index import build_signal_index
from decision_intelligence.models import DecisionDocument
from decision_intelligence.pipeline import Pipeline
ws = Workspace(root=repo)
base = ws.load_documents()
adv = []
for d in cases.D:
    info = DOMAINS[d["domain"]]; verb = {"approved": "approved", "deferred": "deferred", "rejected": "rejected"}[d["status"]]
    adv.append(DecisionDocument(id=d["id"], title=d["title"], domain=d["domain"], subject=d["subject"], date="2026-01-15", status=d["status"],
        owner_role=str(info["owner"]), decision=f"The {info['owner']} {verb} the proposal to {d['title'][0].lower()+d['title'][1:]} for the {info['unit']}.",
        context=d["text"], rationale=""))
docs = base + adv
ents = tuple(entity_names(docs))
comp = Components(documents={d.id: d for d in docs}, chunks=chunk_corpus(docs), embedder=HashingEmbedder(), protocol=ws.load_protocol(),
    signal_indices={"rules-v2": build_signal_index(docs, RuleBasedSignalClassifier(ents))[0], "rules-v3": build_signal_index(docs, ScopedSignalClassifier(ents))[0]},
    entities=ents, scopes=tuple(sorted({d.subject for d in docs} | {d.domain for d in docs})))
checker = make_checker(comp)
def pipe(strategy, gen, check):
    return Pipeline(make_retriever(comp, strategy), make_generator(gen, None, comp.protocol), comp.documents, checker=checker if check else None)
CONF = {"A": ("vector","extractive-v1",False), "B": ("rules-v2","extractive-v1",False), "C": ("rules-v3","extractive-v1",False),
        "E": ("rules-v3","extractive-v1",True), "F": ("rules-v3","extractive-v2",True)}
P = {k: pipe(*v) for k, v in CONF.items()}
def verdict(exp, status, text, cited):
    if status != exp["status"]: return "FAIL"
    t = text.lower()
    if any(c.lower() not in t for c in exp["contains"]): return "FAIL"
    if any(x.lower() in t for x in exp["excludes"]): return "FAIL"
    if exp["docs"] is not None and status == "ANSWER" and set(cited) != set(exp["docs"]): return "FAIL"
    return "PASS"
results = []
for c in cases.C:
    rec = {"id": c["id"], "layer": c["layer"], "question": c["question"], "expect": c["expect"], "why": c["why"]}
    if c["isolated_docs"]:
        r = P["F"].retriever.retrieve(c["question"])
        fake = tuple(dataclasses.replace(r.documents[0], doc_id=i, rank=n+1, admitted=True, gate_reason="audit: forced") for n, i in enumerate(c["isolated_docs"]))
        ans, rep = P["F"].answer(c["question"], dataclasses.replace(r, documents=fake))
        cited = ans.cited_ids()
        rec["isolated"] = {"status": ans.status.value, "text": ans.text, "cited": cited, "sufficiency": rep.to_dict() if rep else None,
                           "verdict": verdict(c["expect"], ans.status.value, ans.text, cited)}
    e2e = {}
    for k, p in P.items():
        res = p.run(c["question"])
        cited = res.answer.cited_ids()
        ranked = [(d.doc_id, round(d.final_score, 3), d.admitted, comp.documents[d.doc_id].subject) for d in res.retrieval.documents]
        e2e[k] = {"status": res.answer.status.value, "text": res.answer.text, "cited": cited, "reason": res.answer.abstain_reason,
                  "top4": ranked, "intent": [ (s.signal.value, s.polarity.value, s.requirement.value) for s in res.retrieval.intent.signals],
                  "sufficiency": res.sufficiency.to_dict() if res.sufficiency else None,
                  "verdict": verdict(c["expect"], res.answer.status.value, res.answer.text, cited)}
    rec["e2e"] = e2e
    results.append(rec)
json.dump(results, open(out_path, "w"), indent=1, default=str)
for r in results:
    iso = r.get("isolated", {}); f = r["e2e"]["F"]
    print(f"{r['id']:5s} exp={r['expect']['status']:7s} | iso={iso.get('status','-'):7s} {iso.get('verdict','-'):4s} | F={f['status']:7s} {f['verdict']:4s} cited={f['cited']} | {(iso.get('sufficiency') or f['sufficiency'] or {}).get('reason','')[:70]}")
