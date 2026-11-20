import re
from dataclasses import dataclass
from typing import Literal, Tuple

_HUMAN_REQUEST = re.compile(
    r"\b(?:i want|i need|please|can you|could you|let me|connect me|transfer me|"
    r"speak (?:to|with)|talk (?:to|with))\b.{0,50}\b"
    r"(?:human|real person|person|agent|representative)\b",
    re.IGNORECASE,
)
_NEGATED_HUMAN_REQUEST = re.compile(
    r"\b(?:i don't want|i do not want|i don't need|i do not need|"
    r"don't connect|do not connect|don't transfer|do not transfer)\b.{0,60}\b"
    r"(?:human|real person|person|agent|representative)\b",
    re.IGNORECASE,
)
_REPORTED_DISTRESS = re.compile(
    r"\b(?:i am|i'm|i feel|i'm feeling|i am feeling)\s+"
    r"(?:(?:really|very|so)\s+)?(?:frustrated|upset|angry|disappointed)\b",
    re.IGNORECASE,
)
_SERVICE_PROBLEM = re.compile(
    r"\b(?:not working|isn't working|doesn't work|still waiting|"
    r"unacceptable|this is ridiculous)\b",
    re.IGNORECASE,
)
_URGENCY = re.compile(
    r"\b(?:urgent|urgently|emergency|immediately|unauthorized charge|"
    r"account compromised|stolen card|fraud)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SupportAssessment:
    """Surface-level support cues; these are not an emotion diagnosis."""

    acknowledgement_recommended: bool
    human_handoff_requested: bool
    human_handoff_recommended: bool
    urgent_language: bool
    signals: Tuple[str, ...]


SupportRoute = Literal[
    "answer",
    "acknowledge_and_answer",
    "human_support",
    "urgent_human_review",
]


@dataclass(frozen=True)
class SupportDecision:
    route: SupportRoute
    offer_human_support: bool
    acknowledge: bool
    clarify_urgency: bool
    reasons: Tuple[str, ...]


def assess_customer_message(message: str) -> SupportAssessment:
    """Identify a small set of explicit wording cues without inferring traits."""
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    human_request = bool(
        _HUMAN_REQUEST.search(message)
        and not _NEGATED_HUMAN_REQUEST.search(message)
    )
    reported_distress = bool(_REPORTED_DISTRESS.search(message))
    service_problem = bool(_SERVICE_PROBLEM.search(message))
    urgent_language = bool(_URGENCY.search(message))

    signals = []
    if human_request:
        signals.append("explicit_human_support_request")
    if reported_distress:
        signals.append("explicitly_reported_distress")
    if service_problem:
        signals.append("service_problem_wording")
    if urgent_language:
        signals.append("urgency_related_wording")

    return SupportAssessment(
        acknowledgement_recommended=reported_distress or service_problem,
        human_handoff_requested=human_request,
        human_handoff_recommended=human_request or urgent_language,
        urgent_language=urgent_language,
        signals=tuple(signals),
    )


def decide_support_action(assessment: SupportAssessment) -> SupportDecision:
    """Choose a support route from explicit wording cues only."""
    if assessment.urgent_language:
        route: SupportRoute = "urgent_human_review"
    elif assessment.human_handoff_requested:
        route = "human_support"
    elif assessment.acknowledgement_recommended:
        route = "acknowledge_and_answer"
    else:
        route = "answer"

    return SupportDecision(
        route=route,
        offer_human_support=assessment.human_handoff_recommended,
        acknowledge=assessment.acknowledgement_recommended,
        clarify_urgency=assessment.urgent_language,
        reasons=assessment.signals,
    )
