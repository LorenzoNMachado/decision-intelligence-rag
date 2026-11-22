"""Deterministic generator of synthetic organisational decision records.

Every record is produced from a seeded pseudo-random generator and a set of
phrase pools written for this project. Organisations, vendors and regulations
are fictional; the audit frameworks named in the compliance pool are public
standards used only as vocabulary.

The generator also records the *intended* label of each semantic signal
(``ground_truth_signals``). Those labels exist so that signal classifiers can be
evaluated; retrieval and generation never read them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from decision_intelligence.models import DecisionDocument, Signal, SignalLabel

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

# (title prefix, verb phrase used inside the decision sentence)
# (title prefix, verb phrase used inside the decision sentence); "{a}" becomes "a"/"an".
ACTIONS: list[tuple[str, str]] = [
    ("Adopt {a}", "adopt {a}"),
    ("Pilot {a}", "run a limited pilot of {a}"),
    ("Expand the", "expand the"),
    ("Standardise the", "standardise the"),
    ("Retire the legacy", "retire the legacy"),
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
        "The work will be delivered by an external provider, {vendor}.",
        "The chosen option depends on {vendor} as the hosting partner.",
        "Delivery relies on a service contract with {vendor}.",
        "{vendor} will operate the service under a managed agreement.",
    ],
    Signal.OPERATIONAL: [
        "The main driver is reducing on-call load and manual toil.",
        "This was motivated by recurring outages and slow incident recovery.",
        "The change targets throughput bottlenecks observed during peak weeks.",
        "Operational efficiency was the primary motivation, with fewer manual handoffs.",
    ],
    Signal.LEGAL: [
        "The change is required to meet obligations under {regulation}.",
        "Counsel confirmed that the process must follow {regulation}.",
        "A statutory deadline under {regulation} makes this change mandatory.",
    ],
    Signal.SECURITY: [
        "It closes an access-control gap found in the last penetration test.",
        "Encryption of data at rest is a precondition for go-live.",
        "The design depends on single sign-on and hardened service credentials.",
        "The rollout is gated on remediation of two vulnerabilities rated high.",
    ],
    Signal.COMPLIANCE: [
        "Evidence produced by this process feeds the annual {framework} audit.",
        "Internal audit placed this control inside the {framework} scope.",
        "The outcome must be documented for {framework} certification.",
    ],
}

ABSENT_PHRASES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: [
        "The team will build and run it in-house, with no outside supplier.",
        "No external vendor is involved; the capability stays internal.",
    ],
    Signal.OPERATIONAL: [
        "Operational concerns were not a factor in this choice.",
        "Day-to-day operations are unaffected and played no part in the decision.",
    ],
    Signal.LEGAL: [
        "There is no legal or regulatory requirement behind this decision.",
        "No statute or regulation applies to this change.",
    ],
    Signal.SECURITY: [
        "The decision has no security implications beyond standard practice.",
        "Security review found nothing that affects this change.",
    ],
    Signal.COMPLIANCE: [
        "This item is out of scope for audit and certification work.",
        "No audit framework covers this activity.",
    ],
}

# Hedged statements: the intended label is UNCLEAR even though signal vocabulary appears.
HEDGED_PHRASES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: ["An external partner may be considered in a later phase."],
    Signal.OPERATIONAL: ["Whether this improves day-to-day operations is still being assessed."],
    Signal.LEGAL: ["Counsel has not yet confirmed whether any regulation applies."],
    Signal.SECURITY: ["A security review is planned but has not started."],
    Signal.COMPLIANCE: ["It is not yet known whether auditors will ask about this process."],
}

CONTEXT_FILLERS = [
    "The current setup was introduced {years} years ago and has grown without a clear owner.",
    "Usage has roughly doubled since the last review.",
    "Two teams maintain overlapping versions of the same capability.",
    "Most of the configuration is undocumented and depends on a few individuals.",
    "Stakeholders from {other} were consulted during discovery.",
]

RATIONALE_FILLERS = [
    "The steering group compared {options} options before choosing this one.",
    "Expected benefits will be reviewed after {months} months.",
    "The estimated effort fits within the current planning cycle.",
    "A fallback plan keeps the previous arrangement available for one quarter.",
]

STATUS_WEIGHTS = [("approved", 0.7), ("deferred", 0.15), ("rejected", 0.15)]


@dataclass(frozen=True)
class CorpusSpec:
    """Size presets: ``demo`` (30 records) and ``experimental`` (150 records)."""

    name: str
    domains: int
    subjects_per_domain: int
    actions_per_subject: int

    @property
    def size(self) -> int:
        return self.domains * self.subjects_per_domain * self.actions_per_subject


SPECS = {
    "demo": CorpusSpec("demo", domains=5, subjects_per_domain=2, actions_per_subject=3),
    "experimental": CorpusSpec("experimental", domains=10, subjects_per_domain=3, actions_per_subject=5),
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


def generate_corpus(spec: CorpusSpec | str = "experimental", seed: int = 2026) -> list[DecisionDocument]:
    """Generate the corpus for a size preset. Same (spec, seed) ⇒ identical output."""

    if isinstance(spec, str):
        spec = SPECS[spec]
    rng = random.Random(seed)
    start = date(2023, 1, 9)
    documents: list[DecisionDocument] = []
    domain_names = list(DOMAINS)[: spec.domains]
    counter = 0
    for domain in domain_names:
        info = DOMAINS[domain]
        subjects = list(info["subjects"])[: spec.subjects_per_domain]  # type: ignore[call-overload]
        for subject in subjects:
            for title_template, verb_template in ACTIONS[: spec.actions_per_subject]:
                title_prefix = title_template.format(a=_article(subject))
                verb = verb_template.format(a=_article(subject))
                counter += 1
                doc_id = f"DEC-{counter:03d}"
                status = _weighted_choice(rng, STATUS_WEIGHTS)
                plan = _signal_plan(rng)
                fills = {
                    "vendor": rng.choice(VENDORS),
                    "regulation": rng.choice(REGULATIONS),
                    "framework": rng.choice(FRAMEWORKS),
                }
                verb_word = {"approved": "approved", "deferred": "deferred", "rejected": "rejected"}[status]
                decision = (
                    f"The {info['owner']} {verb_word} the proposal to {verb} {subject} "
                    f"for the {info['unit']}."
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
                for signal in Signal:
                    sentence = _render_signal(rng, signal, plan[signal], fills)
                    if sentence is None:
                        continue
                    (context_parts if rng.random() < 0.5 else rationale_parts).append(sentence)
                rng.shuffle(context_parts)
                rng.shuffle(rationale_parts)
                metadata = {
                    "cost_band": rng.choice(["low", "medium", "high"]),
                    "risk_level": rng.choice(["low", "moderate", "elevated"]),
                    "review_cycle": rng.choice(["quarterly", "semi-annual", "annual"]),
                }
                # Fills are recorded only when the corresponding sentence was rendered.
                if plan[Signal.THIRD_PARTY] is SignalLabel.PRESENT:
                    metadata["vendor"] = fills["vendor"]
                if plan[Signal.LEGAL] is SignalLabel.PRESENT:
                    metadata["regulation"] = fills["regulation"]
                if plan[Signal.COMPLIANCE] is SignalLabel.PRESENT:
                    metadata["framework"] = fills["framework"]
                documents.append(
                    DecisionDocument(
                        id=doc_id,
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
