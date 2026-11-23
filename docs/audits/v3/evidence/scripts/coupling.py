import json, sys
from decision_intelligence.corpus import generator as g
from decision_intelligence.models import DecisionDocument, Signal
from decision_intelligence.semantic_signals.classifier import RuleBasedSignalClassifier, ScopedSignalClassifier
fills = dict(vendor="Orchard Systems", regulation="the Wage Records Act", regulation_cap="The Wage Records Act", framework="ISO 9001")
def doc(t): return DecisionDocument(id="X", title="Introduce a thing", domain="D", subject="thing", date="2026-01-01", status="approved", owner_role="Owner", decision="", context=t, rationale="")
clf = {"rules-v2": RuleBasedSignalClassifier(), "rules-v3": ScopedSignalClassifier()}
def label(c, sig, text):
    res = clf[c].classify(doc(text)); return str(getattr(res.labels, "get", lambda k, d=None: None)(Signal(sig)) or res.labels[Signal(sig)])
def score(items):
    out = {}
    for c in clf:
        ok = sum(label(c, s, t).endswith(l) for s, l, t in items); out[c] = f"{ok}/{len(items)}"
    return out
gen = [(s.value, "PRESENT", p.format(**fills)) for s, ps in g.PRESENT_PHRASES.items() for p in ps] + \
      [(s.value, "ABSENT", p.format(**fills)) for s, ps in g.ABSENT_PHRASES.items() for p in ps]
MAP={"legal_requirement":"legal_dependency","security_requirement":"security_dependency","compliance_requirement":"compliance_relevance"}
para = [(MAP.get(a,a),b,c) for a,b,c in json.load(open(sys.argv[1]))]
print("generator phrase pool (PRESENT+ABSENT):", len(gen), score(gen))
print("audit paraphrases (labels fixed before run):", len(para), score(para))
for s, l, t in para: print(f"  {l:8s} v2={label('rules-v2', s, t):26s} v3={label('rules-v3', s, t):26s} {s[:12]} | {t}")
