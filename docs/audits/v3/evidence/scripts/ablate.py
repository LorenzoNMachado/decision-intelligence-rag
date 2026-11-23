"""Audit-only ablations on the frozen v3 holdout (analysis; nothing is written to the repo)."""
import json, sys
from collections import Counter
from pathlib import Path
import decision_intelligence.sufficiency.facts as facts
import decision_intelligence.sufficiency.check as check
from decision_intelligence.config import Workspace, Environment, load_components, make_retriever, make_checker, make_generator
from decision_intelligence.pipeline import Pipeline
from decision_intelligence.retrieval.retriever import Retriever
from decision_intelligence.intent.parsers import RuleBasedIntentParser
from decision_intelligence.semantic_signals.lexicon import LEXICON_V2, LEXICON_V3
from decision_intelligence.semantic_signals.scope import Assertion
repo = Path(sys.argv[1]); ws = Workspace(root=repo); comp = load_components(ws, Environment.from_env())
Q = [q for q in ws.load_questions() if q.split == "holdout"]
docs = comp.documents
def run_F():
    p = Pipeline(make_retriever(comp, "rules-v3"), make_generator("extractive-v2", None, comp.protocol), comp.documents, checker=make_checker(comp))
    out = {}
    for q in Q:
        res = p.run(q.text); out[q.id] = res.answer.status.value
    ind = [q for q in Q if q.category in ("entity_missing", "fact_missing")]; ans = [q for q in Q if q.answerable]
    return sum(out[q.id] == "ABSTAIN" for q in ind), sum(out[q.id] == "ABSTAIN" for q in ans)
print("F as frozen: in-domain abstained %d/40, answerable abstained %d/100" % run_F())
orig_m, orig_s = facts._mention_assertion, facts._sentence_assertion
facts._mention_assertion = lambda s, p, e: Assertion.FACTUAL
facts._sentence_assertion = lambda s, e: Assertion.FACTUAL
print("F scope-blind facts (every value FACTUAL): in-domain abstained %d/40, answerable abstained %d/100" % run_F())
orig_types = check._GENERIC_TYPES
check._GENERIC_TYPES = frozenset()  # any single value of the type counts, no attribute-word overlap needed
print("F scope-blind + no attribute focus: in-domain abstained %d/40, answerable abstained %d/100" % run_F())
facts._mention_assertion, facts._sentence_assertion, check._GENERIC_TYPES = orig_m, orig_s, orig_types
# How many in-domain unanswerable targets contain any value of the requested type at all?
TYPE = {"vendor": "ORGANIZATION", "regulation": "REGULATION", "framework": "FRAMEWORK", "budget": "MONEY", "go_live": "DATE", "lead": "PERSON", "users": "QUANTITY"}
c = Counter()
for q in Q:
    if q.category not in ("entity_missing", "fact_missing"): continue
    did, field = q.reference_answer.split(":")[1].split(" states no ")
    did = did.strip(); field = field.strip(" .")
    d = docs[did]; body = f"{d.decision} {d.context} {d.rationale}"
    types = {m.type.value for s in facts.extract_facts(body, comp.entities) for m in s.mentions}
    c[(q.category, field, TYPE[field] in types)] += 1
print("in-domain unanswerable targets: (category, field, record contains a value of the requested type):")
for k, v in sorted(c.items()): print("  ", k, v)
# Retrieval decomposition: parser x signal index (holdout answerable, Recall@4)
ans = [q for q in Q if q.answerable]
def recall(strategy, parser_lex, index_name, ranking):
    params = comp.protocol.retrieval_params(comp.embedder, ranking)
    r = Retriever(strategy=strategy, chunks=comp.chunks, embedder=comp.embedder, params=params,
                  intent_parser=RuleBasedIntentParser(comp.entities, parser_lex), signal_index=comp.signal_indices[index_name], index=make_retriever(comp, "vector").index)
    vals = []
    for q in ans:
        top = r.retrieve(q.text).top_ids()[:4]; vals.append(len(set(q.expected_doc_ids) & set(top)) / len(q.expected_doc_ids))
    return round(sum(vals) / len(vals), 4)
for lex, lname in ((LEXICON_V2, "parser-v2"), (LEXICON_V3, "parser-v3")):
    for idx in ("rules-v2", "rules-v3"):
        for rk in ("flat", "topical"):
            print("Recall@4", lname, "index", idx, rk, recall("rules-v3", lex, idx, rk))
