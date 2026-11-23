"""Independent recomputation of v3 holdout metrics from rows.jsonl + questions.json (stdlib only)."""
import json, math, random, sys
from collections import defaultdict
repo = sys.argv[1]
rows = [json.loads(l) for l in open(f"{repo}/experiments/v3/results/holdout/rows.jsonl")]
Q = {q["id"]: q for q in json.load(open(f"{repo}/data/v3/questions.json"))["questions"]}
summ = json.load(open(f"{repo}/experiments/v3/results/holdout/summary.json"))
by = defaultdict(dict)
for r in rows: by[r["config"]][r["question_id"]] = r
def judge(q, r):
    if not q["answerable"]: return "CORRECT" if r["answer"]["status"] == "ABSTAIN" else "INCORRECT"
    if r["answer"]["status"] != "ANSWER": return "INCORRECT"
    t = r["answer"]["text"].lower()
    found = [f for f in q["key_facts"] if f.lower() in t]; bad = [f for f in q["forbidden_facts"] if f.lower() in t]
    if len(found) == len(q["key_facts"]) and not bad: return "CORRECT"
    return "PARTIAL" if found else "INCORRECT"
def cited(r): return list(dict.fromkeys(s for c in r["answer"]["claims"] for s in c["source_ids"]))
def wilson(k, n, z=1.959964):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (round(c-h,3), round(c+h,3))
def sign_p(a, b):
    n = a+b
    if n == 0: return None
    k = min(a, b); return min(1.0, 2*sum(math.comb(n, i) for i in range(k+1))/2**n)
def boot(diffs, seed=1, B=5000):
    rng = random.Random(seed); n = len(diffs); ms = sorted(sum(rng.choice(diffs) for _ in range(n))/n for _ in range(B))
    return (round(ms[int(0.025*B)],3), round(ms[int(0.975*B)-1],3))
out = {}
for c, rs in sorted(by.items()):
    ans = [(Q[i], r) for i, r in rs.items() if Q[i]["answerable"]]
    una = [(Q[i], r) for i, r in rs.items() if not Q[i]["answerable"]]
    ind = [(q, r) for q, r in una if q["category"] != "out_of_domain"]; ood = [(q, r) for q, r in una if q["category"] == "out_of_domain"]
    rec = [len(set(q["expected_doc_ids"]) & {d["doc_id"] for d in r["retrieved"][:4]})/len(q["expected_doc_ids"]) for q, r in ans]
    prec = []
    for q, r in ans:
        cs = cited(r)
        if r["answer"]["status"] == "ANSWER" and cs: prec.append(sum(x in q["expected_doc_ids"] for x in cs)/len(cs))
    labels = [judge(q, r) for q, r in ans]
    answered = [(q, r) for q, r in rs.items() for q in [Q[q]] if r["answer"]["status"] == "ANSWER"]
    unsup = sum(1 for q, r in answered if not q["answerable"] or not (set(cited(r)) & set(q["expected_doc_ids"])))
    ab = lambda xs: sum(r["answer"]["status"] == "ABSTAIN" for _, r in xs)
    m = dict(n_ans=len(ans), n_ind=len(ind), n_ood=len(ood), recall=sum(rec)/len(rec), precision=sum(prec)/len(prec), n_prec=len(prec),
             strict=labels.count("CORRECT")/len(ans), partial_or_better=(labels.count("CORRECT")+labels.count("PARTIAL"))/len(ans),
             abst_ind=f"{ab(ind)}/{len(ind)}", abst_ood=f"{ab(ood)}/{len(ood)}", false_abst=f"{ab(ans)}/{len(ans)}", unsupported=f"{unsup}/{len(answered)}",
             errors=sum(r["answer"]["status"] == "ERROR" for r in rs.values()))
    s = summ["configs"][c]["lexical_judge"] if "configs" in summ else None
    out[c] = m
    print(c, {k: (round(v,4) if isinstance(v,float) else v) for k, v in m.items()})
# compare with reported summary
print("\nreported vs recomputed (recall, precision, strict):")
for c in out:
    s = summ["configs"][c]["lexical_judge"]
    print(c, round(s["recall_at_k"],4), round(out[c]["recall"],4), "|", round(s["supporting_source_precision"],4), round(out[c]["precision"],4), "|", round(s["correctness"],4), round(out[c]["strict"],4))
def paired(c1, c2, metric):
    better = worse = 0; diffs = []
    for i, q in Q.items():
        if q["split"] != "holdout" or not q["answerable"]: continue
        r1, r2 = by[c1][i], by[c2][i]
        if metric == "recall":
            f = lambda r: len(set(q["expected_doc_ids"]) & {d["doc_id"] for d in r["retrieved"][:4]})/len(q["expected_doc_ids"])
        else:
            f = lambda r: 1.0 if judge(q, r) == "CORRECT" else 0.0
        d = f(r2)-f(r1); diffs.append(d); better += d > 0; worse += d < 0
    return dict(better=better, worse=worse, ties=len(diffs)-better-worse, p=sign_p(better, worse), mean=round(sum(diffs)/len(diffs),3), ci=boot(diffs))
print("\npaired:")
for a, b, m in [("B","C","recall"),("A","G","recall"),("E","F","strict"),("C","E","strict"),("B","D","strict"),("H","F","recall")]:
    print(f"{a}->{b} {m}", paired(a, b, m))
print("\nWilson in-domain 40/40", wilson(40,40), " false abst 14/100", wilson(14,100), " strict F 70/100", wilson(70,100))
# per category strict / recall for B,C,E,F
cats = sorted({q["category"] for q in Q.values()})
print("\nper-category (holdout): recall B/C | strict E/F | abstain E")
for cat in cats:
    ids = [i for i, q in Q.items() if q["split"]=="holdout" and q["category"]==cat]
    def rc(c): 
        v=[len(set(Q[i]["expected_doc_ids"]) & {d["doc_id"] for d in by[c][i]["retrieved"][:4]})/len(Q[i]["expected_doc_ids"]) for i in ids if Q[i]["answerable"]]
        return round(sum(v)/len(v),3) if v else None
    st = lambda c: sum(judge(Q[i], by[c][i])=="CORRECT" for i in ids)
    print(f"{cat:16s} n={len(ids)} recall B={rc('B')} C={rc('C')} | correct E={st('E')} F={st('F')} | ABSTAIN E={sum(by['E'][i]['answer']['status']=='ABSTAIN' for i in ids)}")
