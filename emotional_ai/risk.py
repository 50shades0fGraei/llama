from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

RiskLevel = Literal["low", "medium", "high", "critical"]
AssessmentStatus = Literal["eligible", "human_review", "blocked"]

_RISK_LEVELS = {"low", "medium", "high", "critical"}
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass(frozen=True)
class StakeholderAlignment:
    user: int
    company: int
    service: int


@dataclass(frozen=True)
class SolutionOption:
    name: str
    alignment: StakeholderAlignment
    risk_level: RiskLevel
    constraint_violations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SolutionAssessment:
    option: SolutionOption
    alignment_score: Optional[float]
    status: AssessmentStatus
    rank: Optional[int]


def assess_solution_options(
    options: Sequence[SolutionOption],
) -> Tuple[SolutionAssessment, ...]:
    """Assess explicit options without executing or silently selecting a solution."""
    if isinstance(options, (str, bytes)):
        raise TypeError("options must be a sequence of SolutionOption values")
    if not options:
        raise ValueError("at least one solution option is required")

    names = set()
    scored = []
    for option in options:
        if not isinstance(option, SolutionOption):
            raise TypeError("options must contain SolutionOption values")
        name = " ".join(option.name.split()) if isinstance(option.name, str) else ""
        if not name or len(name) > 120:
            raise ValueError("solution option name must contain 1-120 characters")
        if name.casefold() in names:
            raise ValueError("solution option names must be unique")
        names.add(name.casefold())
        if not isinstance(option.risk_level, str) or option.risk_level not in _RISK_LEVELS:
            raise ValueError("risk_level must be low, medium, high, or critical")
        if not isinstance(option.alignment, StakeholderAlignment):
            raise TypeError("alignment must be a StakeholderAlignment")
        ratings = (
            option.alignment.user,
            option.alignment.company,
            option.alignment.service,
        )
        if any(type(rating) is not int or not 0 <= rating <= 100 for rating in ratings):
            raise ValueError("stakeholder alignment ratings must be integers from 0 to 100")
        if not isinstance(option.constraint_violations, tuple):
            raise TypeError("constraint_violations must be a tuple")
        if any(
            not isinstance(violation, str) or not violation.strip()
            for violation in option.constraint_violations
        ):
            raise ValueError("constraint violations must be non-empty strings")

        blocked = bool(option.constraint_violations)
        if blocked:
            status: AssessmentStatus = "blocked"
        elif option.risk_level in {"high", "critical"}:
            status = "human_review"
        else:
            status = "eligible"
        score = (
            None
            if blocked
            else round(sum(ratings) / 3, 2)
        )
        scored.append((option, score, status))

    eligible = sorted(
        (item for item in scored if item[2] == "eligible"),
        key=lambda item: (
            -item[1],
            _RISK_ORDER[item[0].risk_level],
        ),
    )
    human_review = sorted(
        (item for item in scored if item[2] == "human_review"),
        key=lambda item: (
            -item[1],
            _RISK_ORDER[item[0].risk_level],
        ),
    )
    blocked = [item for item in scored if item[2] == "blocked"]
    ordered = eligible + human_review + blocked
    return tuple(
        SolutionAssessment(
            option=option,
            alignment_score=score,
            status=status,
            rank=(index + 1 if status == "eligible" else None),
        )
        for index, (option, score, status) in enumerate(ordered)
    )
