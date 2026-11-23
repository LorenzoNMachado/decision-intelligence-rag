"""Adversarial cases for the v3 audit. Expectations written BEFORE any system run.
Each doc: (id, title, subject, domain, status, text). Text goes into context; decision line is generic.
expect: status ANSWER/ABSTAIN; contains: strings the answer must contain; excludes: strings it must not contain;
docs: for list questions, the exact set of record ids that should be cited."""
D = []  # documents
C = []  # cases
def doc(i, title, subject, domain, text, status="approved"):
    D.append(dict(id=i, title=title, subject=subject, domain=domain, status=status, text=text))
def case(cid, layer, q, status, contains=(), excludes=(), docs=None, why="", isolated_docs=None):
    C.append(dict(id=cid, layer=layer, question=q, expect=dict(status=status, contains=list(contains), excludes=list(excludes), docs=docs), why=why, isolated_docs=isolated_docs))

# ---------------- Phase B: evidence sufficiency ----------------
doc("ADV-01","Introduce a fleet telematics platform","fleet telematics platform","Logistics",
    "The funding allocation for the programme stands at €410,000. The rollout covers all regional depots.")
case("B01","suff","What budget was approved for the decision to introduce a fleet telematics platform?","ANSWER",["€410,000"],why="synonym: funding allocation = budget",isolated_docs=["ADV-01"])
doc("ADV-02","Introduce a lunch ordering app","lunch ordering app","Workplace Services",
    "Day-to-day responsibility for the rollout sits with Marta Kowalski. The app replaces paper order forms.")
case("B02","suff","Who leads the decision to introduce a lunch ordering app?","ANSWER",["Marta Kowalski"],why="owner described indirectly",isolated_docs=["ADV-02"])
doc("ADV-03","Introduce a parking permit system","parking permit system","Workplace Services","Go-live is scheduled for 03/15/2026. Permits will be issued online.")
doc("ADV-04","Introduce a meeting room sensor network","meeting room sensor network","Workplace Services","The network goes live in spring 2026. Occupancy data is anonymised.")
doc("ADV-05","Introduce a contract clause library","contract clause library","Corporate Finance","Go-live is scheduled for 15.03.2026. Legal staff maintain the clauses.")
case("B03a","suff","When does the decision to introduce a parking permit system go live?","ANSWER",["03/15/2026"],why="US numeric date",isolated_docs=["ADV-03"])
case("B03b","suff","When does the decision to introduce a meeting room sensor network go live?","ANSWER",["spring 2026"],why="season date",isolated_docs=["ADV-04"])
case("B03c","suff","When does the decision to introduce a contract clause library go live?","ANSWER",["15.03.2026"],why="dotted date",isolated_docs=["ADV-05"])
doc("ADV-06","Introduce a supplier risk register","supplier risk register","Corporate Finance",
    "The board rejected a budget of €900,000 for the work. A smaller scope is being prepared.")
case("B04","suff","What budget was approved for the decision to introduce a supplier risk register?","ABSTAIN",excludes=["€900,000"],why="money of a rejected proposal is not an approved budget",isolated_docs=["ADV-06"])
doc("ADV-07","Introduce a field inspection tablet","field inspection tablet","Manufacturing",
    "Orchard Systems was considered during market research, but no contract was signed. The tablets are configured by the plant engineering group.")
case("B05","suff","Which vendor was selected for the decision to introduce a field inspection tablet?","ABSTAIN",excludes=["Orchard Systems"],why="considered, not contracted",isolated_docs=["ADV-07"])
doc("ADV-08","Introduce a returns processing portal","returns processing portal","Retail Operations",
    "Orchard Systems was contracted to run the service. Its competitor Pinecrest Data had also bid for the work.")
case("B06","suff","Which vendor was selected for the decision to introduce a returns processing portal?","ANSWER",["Orchard Systems"],["Pinecrest Data"],why="contracted vs competitor",isolated_docs=["ADV-08"])
doc("ADV-09","Trial a shift swap app","shift swap app","Retail Operations",
    "The trial is planned but has not been approved yet. A budget of €50,000 is planned for it.", status="deferred")
case("B07","suff","What budget was approved for the decision to trial a shift swap app?","ABSTAIN",excludes=["€50,000"],why="planned, not approved",isolated_docs=["ADV-09"])
doc("ADV-10","Introduce a print fleet tracker","print fleet tracker","IT Operations",
    "The change is not required by any law. The Data Retention Act does not apply to printer logs.")
case("B08a","suff","Which regulation does the decision to introduce a print fleet tracker need to comply with?","ABSTAIN",excludes=["Data Retention Act"],why="legal obligation explicitly denied",isolated_docs=["ADV-10"])
case("B08b","suff","Which decisions about the print fleet tracker are driven by a legal obligation?","ABSTAIN",why="legal denied; only record of subject",isolated_docs=["ADV-10"])
doc("ADV-11","Introduce a desk booking tool","desk booking tool","Workplace Services",
    "The compliance team was copied on the announcement. The selected vendor holds ISO 9001 certification.")
case("B09a","suff","Which certification framework does the decision to introduce a desk booking tool support?","ABSTAIN",excludes=["ISO 9001"],why="framework belongs to the vendor, not the decision",isolated_docs=["ADV-11"])
case("B09b","suff","Which decisions about the desk booking tool are needed for audit or certification?","ABSTAIN",why="compliance only as context",isolated_docs=["ADV-11"])
doc("ADV-12","Introduce a mentoring scheme platform","mentoring scheme platform","Talent Management",
    "Sara Lindgren chairs the steering committee. The work is led by Paolo Reyes.")
case("B10a","suff","Who leads the decision to introduce a mentoring scheme platform?","ANSWER",["Paolo Reyes"],["Sara Lindgren"],why="two people, two roles",isolated_docs=["ADV-12"])
case("B10b","suff","Who chairs the steering committee for the decision to introduce a mentoring scheme platform?","ANSWER",["Sara Lindgren"],["Paolo Reyes"],why="two people, two roles",isolated_docs=["ADV-12"])
doc("ADV-13","Introduce a tender evaluation workflow","tender evaluation workflow","Corporate Finance",
    "The contract with Orchard Systems covers hosting. Its annual value is €120,000.")
case("B11a","suff","What is the annual cost of the hosting contract for the decision to introduce a tender evaluation workflow?","ANSWER",["€120,000"],why="value in a sentence that only refers to the contract by 'its'",isolated_docs=["ADV-13"])
case("B11b","suff","Which vendor hosts the service, and at what annual cost, for the decision to introduce a tender evaluation workflow?","ANSWER",["Orchard Systems","€120,000"],why="answer needs two sentences",isolated_docs=["ADV-13"])
doc("ADV-14","Introduce a vehicle damage reporting app","vehicle damage reporting app","Logistics",
    "The approved budget is €200,000. A later board note states that the approved budget is €260,000.")
case("B12","suff","What budget was approved for the decision to introduce a vehicle damage reporting app?","ABSTAIN",why="contradictory values: decline or flag, not pick one",isolated_docs=["ADV-14"])
doc("ADV-15","Introduce a training catalogue","training catalogue","Talent Management",
    "The catalogue lists internal and external courses. Managers approve requests in the tool.")
case("B13","suff","What budget was approved for the decision to introduce a training catalogue?","ABSTAIN",excludes=["€"],why="truly absent",isolated_docs=["ADV-15"])
doc("ADV-16","Introduce an energy metering dashboard","energy metering dashboard","Manufacturing",
    "The budget owner estimates savings of €75,000 per year. Meters report every fifteen minutes.")
case("B14","suff","What budget was approved for the decision to introduce an energy metering dashboard?","ABSTAIN",excludes=["€75,000"],why="right subject, wrong attribute (savings, not budget)",isolated_docs=["ADV-16"])
doc("ADV-17","Introduce a customs declaration service","customs declaration service","Logistics","The approved budget is €330,000. Declarations are filed electronically.")
case("B15a","suff","What budget was approved for the decision to roll out a customs declaration service?","ANSWER",["€330,000"],why="paraphrased reference ('roll out' vs 'introduce')",isolated_docs=["ADV-17"])
case("B15b","e2e","How much money was set aside for the customs declaration service?","ANSWER",["€330,000"],why="different vocabulary, no 'decision to' reference")
doc("ADV-18","Introduce a cost reporting dashboard","cost reporting dashboard","Corporate Finance","All work is done by internal staff, with no outside supplier involved.")
case("B16","suff","Which decisions about the cost reporting dashboard involve no external vendor?","ANSWER",docs=["ADV-18"],why="type noun 'cost' inside the subject name (known defect class)",isolated_docs=["ADV-18"])
doc("ADV-19","Introduce a payroll audit trail","payroll audit trail","Corporate Finance","Without this control the company would be in breach of the Wage Records Act.")
case("B17","suff","Which regulation does the decision to introduce a payroll audit trail need to comply with?","ANSWER",["Wage Records Act"],why="counterfactual legal driver (known defect class)",isolated_docs=["ADV-19"])
doc("ADV-20","Introduce a stock count scanner","stock count scanner","Retail Operations","The work was previously led by Hana Ito; it is now led by Omar Haddad.")
case("B18","suff","Who leads the decision to introduce a stock count scanner?","ANSWER",["Omar Haddad"],["Hana Ito"],why="historical vs current person",isolated_docs=["ADV-20"])
doc("ADV-21","Introduce a lab sample tracker","lab sample tracker","Research","A budget of €60,000 could be approved if the pilot succeeds.")
case("B19","suff","What budget was approved for the decision to introduce a lab sample tracker?","ABSTAIN",excludes=["€60,000"],why="hypothetical only",isolated_docs=["ADV-21"])
doc("ADV-22","Introduce a grant reporting template","grant reporting template","Research","A budget of €220,000 was approved in March. That approval was withdrawn in May after the audit.")
case("B20","suff","What budget was approved for the decision to introduce a grant reporting template?","ABSTAIN",excludes=["€220,000"],why="approval later revoked",isolated_docs=["ADV-22"])
doc("ADV-23","Introduce a recall notification service","recall notification service","Retail Operations","The approved budget for the previous phase was €95,000. Funding of €140,000 has been set aside for this phase.")
case("B21","gen","What budget was approved for the decision to introduce a recall notification service?","ANSWER",["€140,000"],["€95,000"],why="most similar sentence is the wrong (previous) one",isolated_docs=["ADV-23"])
doc("ADV-24","Introduce a visitor badge printer","visitor badge printer","Workplace Services","Orchard Systems hosts the platform. Pinecrest Data provides the support desk under contract.")
case("B22a","gen","Which vendor hosts the platform for the decision to introduce a visitor badge printer?","ANSWER",["Orchard Systems"],["Pinecrest Data"],why="two competing entities",isolated_docs=["ADV-24"])
case("B22b","gen","Which company provides the support desk for the decision to introduce a visitor badge printer?","ANSWER",["Pinecrest Data"],["Orchard Systems"],why="two competing entities",isolated_docs=["ADV-24"])
doc("ADV-25","Introduce a waste collection planner","waste collection planner","Logistics","The project budget is not €500,000 as first reported; it is €450,000.")
case("B23","suff","What budget was approved for the decision to introduce a waste collection planner?","ANSWER",["€450,000"],["€500,000"],why="corrected value in same sentence",isolated_docs=["ADV-25"])
doc("ADV-26","Introduce a cold chain monitor","cold chain monitor","Logistics","Go-live is scheduled for 31 February 2026.")
case("B24","suff","When does the decision to introduce a cold chain monitor go live?","ABSTAIN",why="invalid date",isolated_docs=["ADV-26"])

# ---------------- Phase C: retrieval and list conditions (subject with opposite signals) ----------------
S = "visitor parking system"; Dm = "Workplace Services"
doc("ADV-31","Introduce a visitor parking system",S,Dm,"Orchard Systems hosts the service under contract.")
doc("ADV-32","Trial a visitor parking system",S,Dm,"No external vendor is involved; the trial is run by internal staff.")
doc("ADV-33","Replace the visitor parking system",S,Dm,"The replacement is entirely self-operated by the workplace services team.")
doc("ADV-34","Scale up the visitor parking system",S,Dm,"It is not true that the service runs without an outside provider: Pinecrest Data supplies the barrier hardware.")
doc("ADV-35","Decommission the existing visitor parking system",S,Dm,"No external party is involved, except Pinecrest Data, which removes the old barriers.")
doc("ADV-36","Consolidate the visitor parking system",S,Dm,"An external vendor would be engaged only if the internal team cannot cope with demand.")
case("C01","e2e","Which decisions about the visitor parking system depend on an external vendor?","ANSWER",docs=["ADV-31","ADV-34","ADV-35"],why="explicit, double negation, exception; conditional excluded")
case("C02","e2e","Which decisions about the visitor parking system involve no external vendor?","ANSWER",docs=["ADV-32","ADV-33"],why="explicit and implicit (self-operated) absence")
S2 = "staff canteen ordering app"; D2 = "Workplace Services"
doc("ADV-41","Introduce a staff canteen ordering app",S2,D2,"The change closes a vulnerability found during red-team testing. The app is built and operated by internal staff.")
doc("ADV-42","Trial a staff canteen ordering app",S2,D2,"Members of the security team attended the kick-off meeting. Delivery is outsourced to Orchard Systems.")
doc("ADV-43","Replace the staff canteen ordering app",S2,D2,"Card details were previously stored in plain text; this was fixed last year, and the replacement has no security impact. Internal staff run it.")
doc("ADV-44","Scale up the staff canteen ordering app",S2,D2,"Access will be protected with multi-factor login to stop account takeover. Orchard Systems hosts it.")
case("C03","e2e","Which decisions about the staff canteen ordering app are driven by security concerns?","ANSWER",docs=["ADV-41","ADV-44"],why="synonyms, contextual mention, historical")
case("C04","e2e","Which decisions about the staff canteen ordering app address security and are delivered in-house?","ANSWER",docs=["ADV-41"],why="two constraints")
case("C05","e2e","Which decisions about the staff canteen ordering app depend on an outside company?","ANSWER",docs=["ADV-42","ADV-44"],why="paraphrased condition")
