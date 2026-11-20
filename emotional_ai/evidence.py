from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Evidence:
    source: str
    text: str


class EvidenceRetriever(Protocol):
    """A caller-supplied local or consented source retriever."""

    def search(self, query: str, task: str) -> Sequence[Evidence]:
        ...
