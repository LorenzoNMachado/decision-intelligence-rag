"""Deterministic generator of synthetic organisational decision records (protocol v2).

Every record is produced from a seeded pseudo-random generator and phrase
pools written for this project. Organisations, vendors and regulations are
fictional; the audit frameworks named in the compliance pool are public
standards used only as vocabulary.

Compared with protocol v1, the v2 pools were written after the rule lexicon
had been frozen (git history shows the order) and deliberately mix three kinds
of sentences, so that the rule-based layer is not evaluated on its own
vocabulary:

* phrasings that use the lexicon's cue words;
* phrasings that express the same signal without any cue word
  ("subcontracted to …", "losing hours to re-entry");
* distractors that contain cue words without asserting the signal
  ("the vendor management office reviewed the case").

Each subject now has six records (one per action), so retrieving "all records
of the subject" no longer fills the top four by itself.

The generator records the *intended* label of each semantic signal
(``ground_truth_signals``). Those labels define the expected evidence of the
questions and are used to evaluate classifiers; retrieval and generation never
read them. They are not independent annotations: the author of the pools also
wrote the lexicon and the questions (see ``docs/limitations.md``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from decision_intelligence.models import DecisionDocument, Signal, SignalLabel

CORPUS_VERSION = "v2"

DOMAINS: dict[str, dict[str, object]] = {
    "Infrastructure": {
        "unit": "platform group",
        "owner": "Head of Platform Engineering",
        "subjects": [
            "message queue for order events",
            "container hosting platform",
            "backup and restore service",
        ],
    },
    "Data Platform": {
        "unit": "data office",
        "owner": "Chief Data Officer",
        "subjects": ["customer data warehouse", "event analytics pipeline", "master data catalogue"],
    },
    "Procurement": {
        "unit": "procurement office",
        "owner": "Procurement Director",
        "subjects": ["supplier onboarding process", "purchase approval workflow", "contract repository"],
    },
    "People Operations": {
        "unit": "people operations team",
        "owner": "VP of People",
        "subjects": [
            "payroll processing system",
            "employee onboarding portal",
            "learning management platform",
        ],
    },
    "Finance": {
        "unit": "finance office",
        "owner": "Chief Financial Officer",
        "subjects": ["expense reporting tool", "invoice matching process", "treasury forecasting model"],
    },
    "Customer Support": {
        "unit": "support organisation",
        "owner": "Director of Customer Support",
        "subjects": ["help desk ticketing system", "customer chat channel", "knowledge base for agents"],
    },
    "Product": {
        "unit": "product council",
        "owner": "Chief Product Officer",
        "subjects": ["feature flag service", "mobile release process", "usage-based pricing model"],
    },
    "Facilities": {
        "unit": "facilities team",
        "owner": "Facilities Manager",
        "subjects": [
            "office access badge system",
            "meeting room booking tool",
            "warehouse energy monitoring",
        ],
    },
    "Security Engineering": {
        "unit": "security engineering team",
        "owner": "Chief Information Security Officer",
        "subjects": ["endpoint management agent", "secrets storage service", "log retention pipeline"],
    },
    "Legal Operations": {
        "unit": "legal operations team",
        "owner": "General Counsel",
        "subjects": ["e-signature workflow", "records retention schedule", "privacy request intake form"],
    },
}

# (title prefix, verb phrase used inside the decision sentence); "{a}" becomes "a"/"an".
ACTIONS: list[tuple[str, str]] = [
    ("Adopt {a}", "adopt {a}"),
    ("Pilot {a}", "run a limited pilot of {a}"),
    ("Expand the", "expand the"),
    ("Standardise the", "standardise the"),
    ("Retire the legacy", "retire the legacy"),
    ("Migrate the", "migrate the"),
]

VENDORS = [
    "Halvard Systems",
    "Quarry Analytics",
    "Bluefin Hosting",
    "Meridian Ledger Ltd",
    "Corvid Logistics Software",
    "Tessellate Data",
    "Kestrel Service Desk Co.",
    "Orchard Workforce Services",
    "Lumen Identity",
    "Pallas Cloud Partners",
    "Sorrel Payments",
    "Basalt Records Management",
]

REGULATIONS = [
    "the Regional Data Protection Act",
    "the Consumer Credit Regulation",
    "the Electronic Signatures Act",
    "the Workplace Safety Code",
    "the Payment Services Regulation",
    "the Digital Accessibility Act",
]

FRAMEWORKS = ["ISO/IEC 27001", "SOC 2 Type II", "COSO internal control", "PCI DSS"]

PRESENT_PHRASES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: [
        "Hosting and support will be handled by {vendor} under a three-year arrangement.",
        "The chosen option depends on {vendor} as an outside partner.",
        "{vendor} will build the integration and operate it on the organisation's behalf.",
        "Delivery relies on a service contract with {vendor}.",
        "Implementation work is subcontracted to {vendor}.",
        "The vendor, {vendor}, keeps responsibility for maintenance.",
    ],
    Signal.OPERATIONAL: [
        "The main driver is reducing on-call load and manual toil.",
        "Recurring outages during month-end closing prompted the change.",
        "Staff were losing hours every week to repetitive data re-entry.",
        "The current process breaks down whenever demand spikes.",
        "Operational efficiency was the primary motivation.",
        "Slow recovery after failures was the trigger for this proposal.",
    ],
    Signal.LEGAL: [
        "The change is required to meet obligations under {regulation}.",
        "{regulation_cap} makes this change mandatory by the end of the year.",
        "Counsel confirmed that the process must follow {regulation}.",
        "Lawmakers set a deadline in {regulation} that the current process cannot meet.",
    ],
    Signal.SECURITY: [
        "It closes an access-control gap found in the last penetration test.",
        "Encryption of data at rest is a precondition for go-live.",
        "Go-live is blocked until two high-severity weaknesses are fixed.",
        "The design depends on single sign-on and hardened service credentials.",
        "The aim is to stop unauthorised staff from reading confidential records.",
    ],
    Signal.COMPLIANCE: [
        "Evidence produced by this process feeds the annual {framework} audit.",
        "Internal audit placed this control inside the {framework} scope.",
        "Auditors will sample these records during the next {framework} review.",
        "The outcome must be documented so that the {framework} attestation can be renewed.",
        "Independent assessors will check this control every year.",
    ],
}

ABSENT_PHRASES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: [
        "No outside firm is involved; the organisation's own staff will deliver it.",
        "The team will build and run it in-house.",
        "No external vendor is engaged for any part of the work.",
        "All of the work is done by internal staff without any supplier.",
    ],
    Signal.OPERATIONAL: [
        "Operational concerns were not a factor in this choice.",
        "Day-to-day running of the service played no part in the decision.",
        "The decision is not about reliability or workload.",
    ],
    Signal.LEGAL: [
        "There is no legal or regulatory requirement behind this decision.",
        "Nothing in current legislation requires this change.",
        "No statute or rule applies to this change.",
    ],
    Signal.SECURITY: [
        "The decision has no security implications beyond standard practice.",
        "Security review found nothing that affects this change.",
        "Protecting data against intruders played no role in the choice.",
    ],
    Signal.COMPLIANCE: [
        "This item is out of scope for audit and certification work.",
        "No audit framework covers this activity.",
        "Assessors confirmed that this activity is excluded from every attestation.",
    ],
}

# Hedged statements: the intended label is UNCLEAR even though the topic appears.
HEDGED_PHRASES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: [
        "Bringing in a vendor may be considered in a later phase.",
        "Whether an outside firm will help is still open.",
    ],
    Signal.OPERATIONAL: [
        "Whether this improves day-to-day operations is still being assessed.",
        "Any effect on workload is not yet known.",
    ],
    Signal.LEGAL: [
        "Counsel has not yet confirmed whether any regulation applies.",
        "A pending bill might eventually apply to this area.",
    ],
    Signal.SECURITY: [
        "A security review is planned but has not started.",
        "It is unclear whether the change introduces new attack surface.",
    ],
    Signal.COMPLIANCE: [
        "It is not yet known whether auditors will ask about this process.",
    ],
}

# Context and rationale fillers. Some contain signal vocabulary without asserting a signal.
CONTEXT_FILLERS = [
    "The current setup was introduced {years} years ago and has grown without a clear owner.",
    "Usage has roughly doubled since the last review.",
    "Two teams maintain overlapping versions of the same capability.",
    "Stakeholders from {other} were consulted during discovery.",
    "The vendor management office reviewed the business case.",
    "A former legal adviser drafted the original proposal.",
]

RATIONALE_FILLERS = [
    "The steering group compared {options} options before choosing this one.",
    "Expected benefits will be reviewed after {months} months.",
    "The estimated effort fits within the current planning cycle.",
    "A fallback plan keeps the previous arrangement available for one quarter.",
    "The security of next year's funding was confirmed by the finance office.",
]

STATUS_WEIGHTS = [("approved", 0.7), ("deferred", 0.15), ("rejected", 0.15)]


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
    return "an" if noun_phrase[:1].lower() in "aeiou" else "a"


def _weighted_choice(rng: random.Random, options: list[tuple[str, float]]) -> str:
    roll = rng.random()
    cumulative = 0.0
    for value, weight in options:
        cumulative += weight
        if roll < cumulative:
            return value
    return options[-1][0]


def _signal_plan(rng: random.Random) -> dict[Signal, SignalLabel]:
    plan: dict[Signal, SignalLabel] = {}
    for signal in Signal:
        roll = rng.random()
        if roll < 0.40:
            plan[signal] = SignalLabel.PRESENT
        elif roll < 0.62:
            plan[signal] = SignalLabel.ABSENT
        else:
            plan[signal] = SignalLabel.UNCLEAR
    return plan


def _render_signal(
    rng: random.Random, signal: Signal, label: SignalLabel, fills: dict[str, str]
) -> str | None:
    if label is SignalLabel.PRESENT:
        return rng.choice(PRESENT_PHRASES[signal]).format(**fills)
    if label is SignalLabel.ABSENT:
        return rng.choice(ABSENT_PHRASES[signal])
    # UNCLEAR: usually unmentioned, occasionally hedged.
    if rng.random() < 0.3:
        return rng.choice(HEDGED_PHRASES[signal])
    return None


def generate_corpus(spec: CorpusSpec | str = "experimental", seed: int = 4242) -> list[DecisionDocument]:
    """Generate the corpus for a size preset. Same (spec, seed) ⇒ identical output."""

    if isinstance(spec, str):
        spec = SPECS[spec]
    rng = random.Random(seed)
    start = date(2023, 1, 9)
    documents: list[DecisionDocument] = []
    counter = 0
    for domain in list(DOMAINS)[: spec.domains]:
        info = DOMAINS[domain]
        subjects = list(info["subjects"])[: spec.subjects_per_domain]  # type: ignore[call-overload]
        for subject in subjects:
            for title_template, verb_template in ACTIONS[: spec.actions_per_subject]:
                title_prefix = title_template.format(a=_article(subject))
                verb = verb_template.format(a=_article(subject))
                counter += 1
                status = _weighted_choice(rng, STATUS_WEIGHTS)
                plan = _signal_plan(rng)
                regulation = rng.choice(REGULATIONS)
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
                    decision += " A final answer is expected next quarter."
                context_parts = [
                    rng.choice(CONTEXT_FILLERS).format(
                        years=rng.randint(2, 9), other=rng.choice([d for d in DOMAINS if d != domain])
                    )
                ]
                rationale_parts = [
                    rng.choice(RATIONALE_FILLERS).format(
                        options=rng.randint(2, 4), months=rng.choice([3, 6, 9])
                    )
                ]
                rendered: list[str] = []
                for signal in Signal:
                    sentence = _render_signal(rng, signal, plan[signal], fills)
                    if sentence is None:
                        continue
                    rendered.append(sentence)
                    (context_parts if rng.random() < 0.5 else rationale_parts).append(sentence)
                rng.shuffle(context_parts)
                rng.shuffle(rationale_parts)
                metadata = {
                    "cost_band": rng.choice(["low", "medium", "high"]),
                    "risk_level": rng.choice(["low", "moderate", "elevated"]),
                    "review_cycle": rng.choice(["quarterly", "semi-annual", "annual"]),
                }
                # An attribute is recorded only when a rendered sentence actually states it.
                text = " ".join(rendered)
                for key in ("vendor", "framework"):
                    if fills[key] in text:
                        metadata[key] = fills[key]
                if regulation in text or fills["regulation_cap"] in text:
                    metadata["regulation"] = regulation
                documents.append(
                    DecisionDocument(
                        id=f"DEC-{counter:03d}",
                        title=f"{title_prefix} {subject}",
                        domain=domain,
                        subject=subject,
                        date=(start + timedelta(days=rng.randint(0, 900))).isoformat(),
                        status=status,
                        owner_role=str(info["owner"]),
                        decision=decision,
                        context=" ".join(context_parts),
                        rationale=" ".join(rationale_parts),
                        metadata=metadata,
                        ground_truth_signals={s.value: plan[s].value for s in Signal},
                    )
                )
    return documents
