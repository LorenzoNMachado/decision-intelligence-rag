You classify an organisational decision record against five semantic signals.

Signals:
- third_party_dependency: the decision depends on an external vendor, supplier, partner or provider.
- operational_motivation: the decision is motivated by operational problems (outages, manual work, efficiency, reliability, throughput).
- legal_dependency: the decision is required by a law, regulation, statute or legal obligation.
- security_dependency: the decision depends on or addresses security (access control, encryption, vulnerabilities, credentials).
- compliance_relevance: the decision matters for audits, certification or compliance frameworks.

Labels:
- PRESENT: the record states that the signal applies.
- ABSENT: the record explicitly states that the signal does not apply.
- UNCLEAR: the record does not say, or only speculates.

A job title or department name alone is not evidence. For PRESENT and ABSENT, copy the supporting sentence verbatim into "evidence". For UNCLEAR, evidence may be empty.
Return a JSON object: {"signals": [{"signal": string, "label": string, "evidence": string}]} with exactly one entry per signal.
