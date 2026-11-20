import os
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, Optional, Tuple, Union

QualityTask = Literal[
    "general", "troubleshooting", "policy", "recommendation", "opinion"
]
FeedbackCategory = Literal[
    "too_verbose",
    "too_brief",
    "incorrect",
    "policy_gap",
    "tone_mismatch",
    "offensive_or_unsafe",
    "unresolved",
    "other",
]
FeedbackSource = Literal["user_feedback", "human_review"]
FeedbackSeverity = Literal["low", "high", "critical"]

_TASKS = {"general", "troubleshooting", "policy", "recommendation", "opinion"}
_CATEGORIES = {
    "too_verbose",
    "too_brief",
    "incorrect",
    "policy_gap",
    "tone_mismatch",
    "offensive_or_unsafe",
    "unresolved",
    "other",
}
_SOURCES = {"user_feedback", "human_review"}
_SEVERITIES = {"low", "high", "critical"}
_GUIDANCE = {
    "too_verbose": "Answer the direct question first and omit nonessential detail.",
    "too_brief": "Include the necessary steps, assumptions, and relevant caveats.",
    "incorrect": "Verify claims against available information and state uncertainty.",
    "policy_gap": "Follow supplied policy; do not invent rules or exceptions.",
    "tone_mismatch": "Follow the explicitly selected response style; do not infer emotion.",
    "offensive_or_unsafe": (
        "Use respectful, non-stereotyping language and seek human review when uncertain."
    ),
    "unresolved": "State the next step and do not imply an issue is resolved prematurely.",
}


@dataclass(frozen=True)
class QualityCount:
    task: str
    category: str
    severity: str
    source: str
    count: int


class QualityMemoryStore:
    """Local quality-feedback ledger that stores labels, never conversation text."""

    def __init__(self, path: Union[str, Path]) -> None:
        if not path:
            raise ValueError("path must not be empty")
        self.path = Path(path).expanduser()
        self._initialize()

    @classmethod
    def from_environment(cls) -> Optional["QualityMemoryStore"]:
        """Create a quality store only when explicitly enabled."""
        enabled = os.environ.get("EMOTIONAL_AI_QUALITY_ENABLED", "").strip().casefold()
        if enabled in {"", "0", "false", "no"}:
            return None
        if enabled not in {"1", "true", "yes"}:
            raise ValueError(
                "EMOTIONAL_AI_QUALITY_ENABLED must be 1/true/yes or 0/false/no"
            )
        path = os.environ.get("EMOTIONAL_AI_QUALITY_PATH")
        if not path:
            path = str(Path.home() / ".emotional_ai" / "quality.sqlite3")
        return cls(path)

    def record_feedback(
        self,
        task: str,
        category: str,
        severity: str,
        source: str,
        now: Optional[datetime] = None,
    ) -> None:
        """Record a human- or user-selected label, without storing the response."""
        if task not in _TASKS:
            raise ValueError("unknown support task")
        if category not in _CATEGORIES:
            raise ValueError("unknown feedback category")
        if severity not in _SEVERITIES:
            raise ValueError("severity must be low, high, or critical")
        if source not in _SOURCES:
            raise ValueError("source must be user_feedback or human_review")
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO quality_feedback (task, category, severity, source, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task, category, severity, source, timestamp),
            )

    def summary(self, task: Optional[str] = None) -> Tuple[QualityCount, ...]:
        if task is not None and task not in _TASKS:
            raise ValueError("unknown support task")
        query = """
            SELECT task, category, severity, source, COUNT(*) AS count
            FROM quality_feedback
        """
        parameters = ()
        if task is not None:
            query += " WHERE task = ?"
            parameters = (task,)
        query += " GROUP BY task, category, severity, source ORDER BY task, category"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            QualityCount(
                task=row["task"],
                category=row["category"],
                severity=row["severity"],
                source=row["source"],
                count=row["count"],
            )
            for row in rows
        )

    def guidance_for(self, task: str, min_reviews: int = 3) -> Tuple[str, ...]:
        """Return prompt hints only from repeated human-reviewed feedback."""
        if task not in _TASKS:
            raise ValueError("unknown support task")
        if type(min_reviews) is not int or min_reviews < 1:
            raise ValueError("min_reviews must be a positive integer")

        reviewed = [
            item
            for item in self.summary(task)
            if item.source == "human_review"
        ]
        critical = [
            item for item in reviewed if item.severity == "critical"
        ]
        guidance = []
        if critical:
            guidance.append(
                "A critical issue was human-reviewed for this task. Be conservative "
                "and recommend human review when uncertain; do not change policy."
            )
        for item in reviewed:
            if item.category in _GUIDANCE and (
                item.count >= min_reviews or item.severity == "critical"
            ):
                hint = _GUIDANCE[item.category]
                if hint not in guidance:
                    guidance.append(hint)
        return tuple(guidance)

    def clear(self) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM quality_feedback")

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_feedback (
                    id INTEGER PRIMARY KEY,
                    task TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as connection:
            with connection:
                yield connection

    @staticmethod
    def _timestamp(now: Optional[datetime]) -> str:
        value = datetime.now(timezone.utc) if now is None else now
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat()
