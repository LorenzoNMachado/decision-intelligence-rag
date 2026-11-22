"""Evidence sufficiency (protocol v3): does the admitted evidence contain what is asked?"""

from decision_intelligence.semantic_signals.scope import Assertion
from decision_intelligence.sufficiency.check import (
    DocumentSufficiency,
    EvidenceSufficiencyChecker,
    Sufficiency,
    SufficiencyReport,
)
from decision_intelligence.sufficiency.facts import AnswerType, SentenceFacts, TypedMention, extract_facts
from decision_intelligence.sufficiency.need import InformationNeed, analyse_need

__all__ = [
    "AnswerType",
    "Assertion",
    "DocumentSufficiency",
    "EvidenceSufficiencyChecker",
    "InformationNeed",
    "SentenceFacts",
    "Sufficiency",
    "SufficiencyReport",
    "TypedMention",
    "analyse_need",
    "extract_facts",
]
