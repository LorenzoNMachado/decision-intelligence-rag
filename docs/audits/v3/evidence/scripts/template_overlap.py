import json, re
from collections import defaultdict
Q=json.load(open("data/v3/questions.json"))["questions"]; S=json.load(open("data/v3/question_specs.json"))["categories"]
docs={json.loads(l)["id"]:json.loads(l) for l in open("data/v3/corpus.jsonl")}
def tmpl_of(q):
    cat=q["category"]; spec=S[cat]; ts=S[spec["templates_from"]]["templates"] if "templates_from" in spec else spec.get("templates",[])
    for t in ts:
        pat=re.escape(t["text"]).replace(r"\{title\}","(.+)").replace(r"\{subject\}","(.+)")
        if re.fullmatch(pat,q["text"]): return t["text"]
    return "ood:"+q["text"]
split=defaultdict(lambda: defaultdict(set))
for q in Q: split[q["category"]][q["split"]].add(tmpl_of(q))
tot=shared=0
for cat,d in split.items():
    h=d["holdout"]; dv=d["dev"]; s=len(h&dv); tot+=len(h); shared+=s
    print(f"{cat:16s} holdout templates {len(h):2d}, also used in dev {s:2d}")
hq=[q for q in Q if q["split"]=="holdout" and q["category"]!="out_of_domain"]
dev_t={tmpl_of(q) for q in Q if q["split"]=="dev"}
print("holdout in-domain questions whose phrasing also appears in dev:", sum(tmpl_of(q) in dev_t for q in hq), "/", len(hq))
