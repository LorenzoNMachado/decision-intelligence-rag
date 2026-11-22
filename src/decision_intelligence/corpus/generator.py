"""Deterministic generator of synthetic organisational decision records (protocol v3).

Written after the v3 system (sufficiency check, assertion scope, lexicon) was
committed (53dbec6), with new domains, subjects, vendors, people and phrase
pools that share no sentence template with the v1/v2 generators. Records are
longer and mix, for each of the five semantic signals:

* current statements, with and without the lexicon's cue words, including
  dependencies stated only through a named organisation;
* explicit and implicit negations;
* hypothetical statements (intended label UNCLEAR);
* historical contrasts, where a former state differs from the current one
  (the intended label is the *current* state);
* contextual distractors that use a cue word without asserting the signal.

Each record may also state four attributes — an approved budget, a go-live
date, a lead and a number of users — or only a near miss for them (a former
contract value, a planned review date, a consulted person, a historical
user count). The generator records the intended signal labels and the
attributes it actually stated (``ground_truth_signals``, ``metadata``); those
are the only source of expected answers and are never read by the system.

Independence is limited: the same author wrote this generator and the system.
The generator was written after the system was frozen, but not by an
independent annotator (``docs/v3/protocol.md``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from decision_intelligence.models import DecisionDocument, Signal, SignalLabel

CORPUS_VERSION = "v3"

DOMAINS: dict[str, dict[str, object]] = {
    "Clinical Services": {
        "unit": "clinical operations group",
        "owner": "Director of Clinical Services",
        "subjects": ["patient scheduling system", "lab results portal", "pharmacy stock tracker"],
    },
    "Manufacturing": {
        "unit": "plant engineering group",
        "owner": "Plant Director",
        "subjects": ["production line sensors", "quality inspection workflow", "spare parts catalogue"],
    },
    "Retail Operations": {
        "unit": "store operations office",
        "owner": "Head of Retail Operations",
        "subjects": ["point-of-sale terminals", "loyalty programme platform", "store staffing planner"],
    },
    "Logistics": {
        "unit": "distribution planning group",
        "owner": "Logistics Director",
        "subjects": ["route optimisation engine", "warehouse scanning devices", "carrier booking portal"],
    },
    "Talent Management": {
        "unit": "talent office",
        "owner": "Chief People Officer",
        "subjects": ["recruitment tracking system", "performance review cycle", "employee benefits portal"],
    },
    "Corporate Finance": {
        "unit": "group finance office",
        "owner": "Group Financial Controller",
        "subjects": ["accounts payable workflow", "budget planning tool", "corporate card programme"],
    },
    "IT Operations": {
        "unit": "infrastructure services group",
        "owner": "Head of IT Operations",
        "subjects": ["service monitoring stack", "identity directory", "laptop provisioning process"],
    },
    "Marketing": {
        "unit": "brand and campaigns office",
        "owner": "Marketing Director",
        "subjects": ["customer email platform", "campaign analytics dashboard", "brand asset library"],
    },
    "Research": {
        "unit": "research support office",
        "owner": "Head of Research Support",
        "subjects": ["electronic lab notebook", "research data repository", "grant tracking database"],
    },
    "Workplace Services": {
        "unit": "workplace services team",
        "owner": "Workplace Services Manager",
        "subjects": ["travel booking process", "document archive", "visitor registration kiosk"],
    },
}

ACTIONS: list[tuple[str, str]] = [
    ("Introduce {a}", "introduce {a}"),
    ("Trial {a}", "trial {a}"),
    ("Scale up the", "scale up the"),
    ("Consolidate the", "consolidate the"),
    ("Decommission the existing", "decommission the existing"),
    ("Replace the", "replace the"),
]

VENDORS = [
    "Arden Field Systems",
    "Brightwater Logistics Software",
    "Cobalt Ridge Analytics",
    "Delmar Health Informatics",
    "Eastgate Retail Technologies",
    "Fernhill Cloud Services",
    "Greyhaven Identity Labs",
    "Harbourline Payments",
    "Ivory Lane Consulting",
    "Juniper Point Hosting",
    "Kestrel Bay Data",
    "Larkspur Facility Services",
]
PEOPLE = [
    "Amara Okafor",
    "Bruno Castellan",
    "Chiara Lindqvist",
    "Dmitri Halloran",
    "Elif Sandoval",
    "Farid Nakamura",
    "Greta Holloway",
    "Hugo Asante",
    "Ines Marchetti",
    "Jonas Whitcombe",
    "Keiko Brannigan",
    "Lucia Ferreira",
]
REGULATIONS = [
    "the Patient Records Integrity Act",
    "the Fair Trading Disclosure Regulation",
    "the Workplace Accessibility Code",
    "the Digital Identity Assurance Act",
    "the Freight Safety Regulation",
    "the Research Data Stewardship Act",
]
FRAMEWORKS = ["ISO/IEC 27001", "SOC 2 Type II", "PCI DSS", "ISO 9001"]
AMOUNTS = [
    "€85,000",
    "€120,000",
    "€240,000",
    "€310,000",
    "€475,000",
    "€650,000",
    "€1.2 million",
    "€2.4 million",
]
USER_COUNTS = ["150", "420", "860", "1,200", "2,300", "4,800"]

PRESENT_PHRASES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: [
        "Day-to-day operation will rely on {vendor}, which runs the service under contract.",
        "{vendor} was selected as the external supplier for the build.",
        "The configuration and support are outsourced to {vendor}.",
        "A service agreement with {vendor} covers hosting and maintenance.",
        "The platform itself is provided by {vendor}.",
    ],
    Signal.OPERATIONAL: [
        "Repeated outages in the last two quarters forced the change.",
        "Teams spend several days each month fixing the same manual errors.",
        "The present setup cannot keep up with peak demand.",
        "The goal is to remove handoffs that slow down every request.",
        "Recovery after a failure currently takes most of a working day.",
    ],
    Signal.LEGAL: [
        "{regulation_cap} requires the change before the next reporting period.",
        "Without the change the organisation would breach {regulation}.",
        "The work is a legal obligation under {regulation}.",
        "Compliance with {regulation} is the reason for the timing.",
    ],
    Signal.SECURITY: [
        "Encryption of stored records is a condition of approval.",
        "The change removes shared administrator credentials.",
        "A penetration test found weaknesses that this change closes.",
        "Access will be restricted through single sign-on.",
    ],
    Signal.COMPLIANCE: [
        "The resulting records will be sampled in the {framework} audit.",
        "Auditors asked for this control to support {framework} certification.",
        "The process becomes part of the evidence for {framework}.",
        "Internal audit has added the new control to its annual plan.",
    ],
}

ABSENT_PHRASES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: [
        "All work will be done by internal staff, with no external supplier.",
        "The solution is built and run entirely in-house.",
        "The service remains free of any vendor dependency.",
    ],
    Signal.OPERATIONAL: [
        "Operational issues played no part in the decision.",
        "The current service runs without incidents, so reliability was not a factor.",
    ],
    Signal.LEGAL: [
        "No law or regulation requires this change.",
        "The decision is not driven by any legal obligation.",
    ],
    Signal.SECURITY: [
        "The change has no security implications.",
        "Security was reviewed and found not to be affected.",
    ],
    Signal.COMPLIANCE: [
        "The activity is excluded from audit work.",
        "No certification or audit relies on this process.",
    ],
}

HYPOTHETICAL_PHRASES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: ["If the trial succeeds, an external vendor could be engaged later."],
    Signal.OPERATIONAL: ["The change might ease operational pressure, but that has not been measured."],
    Signal.LEGAL: ["A proposed regulation may apply in the future."],
    Signal.SECURITY: ["A security assessment could be requested if the scope grows."],
    Signal.COMPLIANCE: ["Whether auditors will examine the process is still being discussed."],
}

# Former state differs from the current one: (current label, sentence).
HISTORICAL_CONTRASTS: dict[Signal, list[tuple[SignalLabel, str]]] = {
    Signal.THIRD_PARTY: [
        (SignalLabel.ABSENT, "Until last year the service relied on {vendor}; it is now run in-house."),
        (SignalLabel.PRESENT, "The service used to be handled internally; it is now operated by {vendor}."),
    ],
    Signal.LEGAL: [
        (
            SignalLabel.ABSENT,
            "A former regulation once applied to this area, but no legal requirement remains.",
        ),
    ],
    Signal.COMPLIANCE: [
        (
            SignalLabel.ABSENT,
            "The process was previously in audit scope; it is now excluded from audit work.",
        ),
    ],
}

DISTRACTORS = [
    "The vendor management office reviewed the business case.",
    "A member of the security team attended the kick-off meeting.",
    "The audit committee received a short summary for information.",
    "A former legal adviser commented on an early draft.",
    "The operations desk will be told about the change by email.",
]
FILLERS = [
    "The current arrangement dates back {years} years.",
    "Usage roughly doubled after the last reorganisation.",
    "Several teams maintain their own spreadsheets for the same task.",
    "Colleagues from {other} were consulted during the review.",
    "The steering group compared {options} options before deciding.",
    "Benefits will be reviewed after {months} months.",
    "The effort fits within the current planning cycle.",
    "A fallback keeps the previous arrangement available for one quarter.",
]

BUDGET_PHRASES = [
    "The approved budget is {amount}.",
    "A budget of {amount} was approved for the first year.",
    "Funding of {amount} has been set aside for the work.",
    "The project budget is capped at {amount}.",
]
BUDGET_NEAR = [
    "The previous contract was worth {amount}.",
    "A further {amount} may be requested if the scope grows.",
]
GO_LIVE_PHRASES = [
    "Go-live is scheduled for {date}.",
    "The new arrangement goes live on {date}.",
    "The rollout will be complete and live by {date}.",
]
GO_LIVE_NEAR = ["A progress review is planned for {date}."]
LEAD_PHRASES = [
    "The work is led by {person}.",
    "{person} is accountable for delivery.",
    "{person}, the project lead, reports progress monthly.",
]
LEAD_NEAR = ["{person} from {other} was consulted on the requirements."]
USERS_PHRASES = [
    "About {users} employees will use the new system.",
    "The change affects roughly {users} staff.",
    "Around {users} people will use it every week.",
]
USERS_NEAR = ["The old setup was used by around {users} people until last year."]

STATUS_WEIGHTS = [("approved", 0.7), ("deferred", 0.15), ("rejected", 0.15)]
ATTRIBUTE_PROBABILITY = 0.45
NEAR_MISS_PROBABILITY = 0.5


@dataclass(frozen=True)
class CorpusSpec:
    """Size presets: ``demo`` (36 records) and ``experimental`` (180 records)."""

    name: str
    domains: int
    subjects_per_domain: int
    actions_per_subject: int

    @property
    def size(self) -> int:
        return self.domains * self.subjects_per_domain * self.actions_per_subject


SPECS = {
    "demo": CorpusSpec("demo", domains=3, subjects_per_domain=2, actions_per_subject=6),
    "experimental": CorpusSpec("experimental", domains=10, subjects_per_domain=3, actions_per_subject=6),
}


def _article(noun_phrase: str) -> str:
    return "an" if noun_phrase[:1].lower() in "aeio" else "a"


def _weighted_choice(rng: random.Random, options: list[tuple[str, float]]) -> str:
    roll = rng.random()
    cumulative = 0.0
    for value, weight in options:
        cumulative += weight
        if roll < cumulative:
            return value
    return options[-1][0]


def _date(rng: random.Random) -> str:
    day = date(2025, 1, 1) + timedelta(days=rng.randint(0, 900))
    return f"{day.day} {day:%B} {day.year}"


def _signal_sentences(
    rng: random.Random, signal: Signal, fills: dict[str, str]
) -> tuple[SignalLabel, list[str], bool]:
    """(intended label, sentences, whether a named vendor is stated as current)."""

    roll = rng.random()
    contrasts = HISTORICAL_CONTRASTS.get(signal, [])
    if contrasts and roll < 0.12:
        label, sentence = rng.choice(contrasts)
        current_vendor = label is SignalLabel.PRESENT and "{vendor}" in sentence
        return label, [sentence.format(**fills)], current_vendor
    if roll < 0.45:
        sentence = rng.choice(PRESENT_PHRASES[signal])
        return SignalLabel.PRESENT, [sentence.format(**fills)], "{vendor}" in sentence
    if roll < 0.65:
        return SignalLabel.ABSENT, [rng.choice(ABSENT_PHRASES[signal])], False
    if roll < 0.78:
        return SignalLabel.UNCLEAR, [rng.choice(HYPOTHETICAL_PHRASES[signal])], False
    return SignalLabel.UNCLEAR, [], False


def generate_corpus(spec: CorpusSpec | str = "experimental", seed: int = 7373) -> list[DecisionDocument]:
    """Generate the corpus for a size preset. Same (spec, seed) ⇒ identical output."""

    if isinstance(spec, str):
        spec = SPECS[spec]
    rng = random.Random(seed)
    start = date(2024, 1, 8)
    documents: list[DecisionDocument] = []
    counter = 0
    for domain in list(DOMAINS)[: spec.domains]:
        info = DOMAINS[domain]
        subjects = list(info["subjects"])[: spec.subjects_per_domain]  # type: ignore[call-overload]
        for subject in subjects:
            for title_template, verb_template in ACTIONS[: spec.actions_per_subject]:
                counter += 1
                title = f"{title_template.format(a=_article(subject))} {subject}"
                verb = verb_template.format(a=_article(subject))
                status = _weighted_choice(rng, STATUS_WEIGHTS)
                regulation = rng.choice(REGULATIONS)
                other = rng.choice([d for d in DOMAINS if d != domain])
                fills = {
                    "vendor": rng.choice(VENDORS),
                    "regulation": regulation,
                    "regulation_cap": regulation[:1].upper() + regulation[1:],
                    "framework": rng.choice(FRAMEWORKS),
                }
                decision = (
                    f"The {info['owner']} {status} the proposal to {verb} {subject} for the {info['unit']}."
                )
                if status == "deferred":
                    decision += " A final answer is expected after the next planning round."
                context = [
                    rng.choice(FILLERS).format(
                        years=rng.randint(3, 12),
                        other=other,
                        options=rng.randint(2, 4),
                        months=rng.choice([3, 6]),
                    )
                ]
                rationale = [
                    rng.choice(FILLERS[4:]).format(options=rng.randint(2, 4), months=rng.choice([3, 6, 9]))
                ]
                if rng.random() < 0.6:
                    context.append(rng.choice(DISTRACTORS))
                labels: dict[str, str] = {}
                metadata: dict[str, str] = {}
                for signal in Signal:
                    label, sentences, vendor_stated = _signal_sentences(rng, signal, fills)
                    labels[signal.value] = label.value
                    for sentence in sentences:
                        (context if rng.random() < 0.5 else rationale).append(sentence)
                    if vendor_stated:
                        metadata["vendor"] = fills["vendor"]
                    text = " ".join(sentences)
                    if (
                        signal is Signal.LEGAL
                        and label is SignalLabel.PRESENT
                        and regulation.removeprefix("the ") in text
                    ):
                        metadata["regulation"] = regulation
                    if (
                        signal is Signal.COMPLIANCE
                        and label is SignalLabel.PRESENT
                        and fills["framework"] in text
                    ):
                        metadata["framework"] = fills["framework"]
                attributes = [
                    ("budget", BUDGET_PHRASES, BUDGET_NEAR, {"amount": rng.choice(AMOUNTS)}),
                    ("go_live", GO_LIVE_PHRASES, GO_LIVE_NEAR, {"date": _date(rng)}),
                    ("lead", LEAD_PHRASES, LEAD_NEAR, {"person": rng.choice(PEOPLE), "other": other}),
                    ("users", USERS_PHRASES, USERS_NEAR, {"users": rng.choice(USER_COUNTS)}),
                ]
                for key, phrases, near, values in attributes:
                    if rng.random() < ATTRIBUTE_PROBABILITY:
                        rationale.append(rng.choice(phrases).format(**values))
                        metadata[key] = next(iter(values.values()))
                    elif rng.random() < NEAR_MISS_PROBABILITY:
                        # Near misses use a different value, so a wrong answer is detectable.
                        miss = dict(values)
                        if key == "budget":
                            miss["amount"] = rng.choice([a for a in AMOUNTS if a != values["amount"]])
                        elif key == "go_live":
                            miss["date"] = _date(rng)
                        elif key == "lead":
                            miss["person"] = rng.choice([p for p in PEOPLE if p != values["person"]])
                        else:
                            miss["users"] = rng.choice([u for u in USER_COUNTS if u != values["users"]])
                        context.append(rng.choice(near).format(**miss))
                rng.shuffle(context)
                rng.shuffle(rationale)
                metadata.update(
                    {
                        "cost_band": rng.choice(["low", "medium", "high"]),
                        "risk_level": rng.choice(["low", "moderate", "elevated"]),
                    }
                )
                documents.append(
                    DecisionDocument(
                        id=f"DEC-{counter:03d}",
                        title=title,
                        domain=domain,
                        subject=subject,
                        date=(start + timedelta(days=rng.randint(0, 700))).isoformat(),
                        status=status,
                        owner_role=str(info["owner"]),
                        decision=decision,
                        context=" ".join(context),
                        rationale=" ".join(rationale),
                        metadata=metadata,
                        ground_truth_signals=labels,
                    )
                )
    return documents
