from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class InterestPriority:
    interest: str
    rank: int


def rank_user_stated_interests(
    interests: Iterable[str], *, consented: bool
) -> Tuple[InterestPriority, ...]:
    """Rank interests in the order the user explicitly supplied them.

    This function does not infer, discover, or persist preferences. The caller is
    responsible for obtaining consent and passing only user-stated interests.
    """
    if type(consented) is not bool:
        raise TypeError("consented must be bool")
    if not consented:
        raise PermissionError("explicit consent is required to rank interests")
    if isinstance(interests, str):
        raise TypeError("interests must be a collection of labels, not one string")

    ranked = []
    seen = set()
    for interest in interests:
        if not isinstance(interest, str):
            raise TypeError("each interest must be a string")
        normalized = interest.strip()
        if not normalized:
            raise ValueError("interests must not be empty")
        key = normalized.casefold()
        if key in seen:
            raise ValueError("interests must be unique")
        seen.add(key)
        ranked.append(InterestPriority(interest=normalized, rank=len(ranked) + 1))

    return tuple(ranked)
