from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

GoalSource = Literal["user_explicit", "task", "human_review"]

_GOAL_SOURCES = {"user_explicit", "task", "human_review"}
_MAX_GOALS = 8
_MAX_GOAL_LENGTH = 120

CORE_GOAL_HIERARCHY = (
    "Follow applicable law and system policy, and protect user safety; these are non-overridable.",
    "Be truthful, ground claims in evidence, state uncertainty, and never claim "
    "unverified actions.",
    "Protect privacy and consent, preserve user agency, and avoid coercion.",
    "Within those safeguards, consider explicit user goals, documented company "
    "goals, and service reliability as parallel interests.",
    "Address the stated request and apply caller-supplied priorities only within the fixed safeguards.",
    "Optimize tone, detail, and efficiency only after higher priorities are met.",
)


@dataclass(frozen=True)
class GoalPriority:
    goal: str
    weight: int
    source: GoalSource


def rank_goal_priorities(
    priorities: Sequence[GoalPriority],
) -> Tuple[GoalPriority, ...]:
    """Validate and order explicit adaptive goals; weights never override core rules."""
    if isinstance(priorities, (str, bytes)):
        raise TypeError("priorities must be a sequence of GoalPriority values")
    if len(priorities) > _MAX_GOALS:
        raise ValueError(f"at most {_MAX_GOALS} adaptive goals may be supplied")

    normalized = []
    seen = set()
    for item in priorities:
        if not isinstance(item, GoalPriority):
            raise TypeError("priorities must contain GoalPriority values")
        if not isinstance(item.goal, str):
            raise TypeError("goal must be a string")
        goal = " ".join(item.goal.split())
        if not goal or len(goal) > _MAX_GOAL_LENGTH:
            raise ValueError(
                f"goal must contain 1-{_MAX_GOAL_LENGTH} characters"
            )
        key = goal.casefold()
        if key in seen:
            raise ValueError("adaptive goals must be unique")
        seen.add(key)
        if type(item.weight) is not int or not 1 <= item.weight <= 100:
            raise ValueError("goal weight must be an integer from 1 to 100")
        if item.source not in _GOAL_SOURCES:
            raise ValueError(
                "goal source must be user_explicit, task, or human_review"
            )
        normalized.append(
            GoalPriority(goal=goal, weight=item.weight, source=item.source)
        )
    return tuple(sorted(normalized, key=lambda item: -item.weight))
