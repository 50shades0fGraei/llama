import json
import os
import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Union

_ENABLED_ENV = "EMOTIONAL_AI_MEMORY_ENABLED"
_PATH_ENV = "EMOTIONAL_AI_MEMORY_PATH"
_ENABLED_VALUES = {"1", "true", "yes"}
_DISABLED_VALUES = {"", "0", "false", "no"}
_SENSITIVE_LABEL = re.compile(
    r"\b(?:password|passcode|pin|cvv|cvc|ssn|social security|api key|"
    r"access token|private key|credit card number|bank account number)\b"
    r"|\b(?:\d[ -]?){13,19}\b"
    r"|[$€£]\s?\d+(?:[.,]\d{2})?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Reminder:
    id: str
    topic: str
    keywords: Tuple[str, ...]
    due_on: Optional[date]
    created_at: datetime
    last_referenced_at: datetime


class MemoryStore:
    """Local, profile-scoped reminder storage; no conversation text is retained."""

    def __init__(self, path: Union[str, Path], inactivity_days: int = 90) -> None:
        if not path:
            raise ValueError("path must not be empty")
        if type(inactivity_days) is not int or inactivity_days < 1:
            raise ValueError("inactivity_days must be a positive integer")
        self.path = Path(path).expanduser()
        self.inactivity_days = inactivity_days
        self._initialize()

    @classmethod
    def from_environment(cls) -> Optional["MemoryStore"]:
        """Create a store only when explicitly enabled in the environment."""
        enabled = os.environ.get(_ENABLED_ENV, "").strip().casefold()
        if enabled in _DISABLED_VALUES:
            return None
        if enabled not in _ENABLED_VALUES:
            raise ValueError(
                f"{_ENABLED_ENV} must be one of 1/true/yes or 0/false/no"
            )
        path = os.environ.get(_PATH_ENV)
        if not path:
            path = str(Path.home() / ".emotional_ai" / "memory.sqlite3")
        return cls(path)

    def add_reminder(
        self,
        topic: str,
        keywords: Iterable[str],
        due_on: Optional[date] = None,
        *,
        consented: bool,
        now: Optional[datetime] = None,
    ) -> Reminder:
        """Save compact reminder metadata after the user has opted in."""
        if type(consented) is not bool:
            raise TypeError("consented must be bool")
        if not consented:
            raise PermissionError("user consent is required to save a reminder")

        clean_topic = self._clean_label(topic, "topic")
        clean_keywords = self._clean_keywords(keywords)
        if due_on is not None and (
            not isinstance(due_on, date) or isinstance(due_on, datetime)
        ):
            raise TypeError("due_on must be a date or None")

        timestamp = self._timestamp(now)
        reminder = Reminder(
            id=str(uuid.uuid4()),
            topic=clean_topic,
            keywords=clean_keywords,
            due_on=due_on,
            created_at=datetime.fromisoformat(timestamp),
            last_referenced_at=datetime.fromisoformat(timestamp),
        )
        with self._connection() as connection:
            self._delete_expired(connection, timestamp)
            connection.execute(
                """
                INSERT INTO reminders
                    (id, topic, keywords, due_on, created_at, last_referenced_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    reminder.id,
                    reminder.topic,
                    json.dumps(reminder.keywords),
                    reminder.due_on.isoformat() if reminder.due_on else None,
                    timestamp,
                    timestamp,
                ),
            )
        return reminder

    def list_open(self, now: Optional[datetime] = None) -> Tuple[Reminder, ...]:
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            self._delete_expired(connection, timestamp)
            rows = connection.execute(
                """
                SELECT id, topic, keywords, due_on, created_at, last_referenced_at
                FROM reminders
                ORDER BY due_on IS NULL, due_on, created_at
                """
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def find_relevant(
        self, message: str, now: Optional[datetime] = None
    ) -> Tuple[Reminder, ...]:
        """Return open reminders whose saved topic or keywords occur in the message."""
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        query = self._normalize_phrase(message)
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            self._delete_expired(connection, timestamp)
            rows = connection.execute(
                """
                SELECT id, topic, keywords, due_on, created_at, last_referenced_at
                FROM reminders
                """
            ).fetchall()
            matches = []
            for row in rows:
                phrases = (row["topic"], *json.loads(row["keywords"]))
                if any(self._contains_phrase(query, phrase) for phrase in phrases):
                    matches.append(row)
            connection.executemany(
                "UPDATE reminders SET last_referenced_at = ? WHERE id = ?",
                ((timestamp, row["id"]) for row in matches),
            )
        return tuple(
            self._from_row(row, last_referenced_at=timestamp) for row in matches
        )

    def resolve(self, reminder_id: str) -> None:
        """Remove a completed reminder immediately."""
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM reminders WHERE id = ?", (reminder_id,)
            )
            if cursor.rowcount == 0:
                raise KeyError(f"no open reminder with id {reminder_id!r}")

    def clear(self) -> None:
        """Delete all reminders in this profile's store."""
        with self._connection() as connection:
            connection.execute("DELETE FROM reminders")

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    due_on TEXT,
                    created_at TEXT NOT NULL,
                    last_referenced_at TEXT NOT NULL
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

    def _delete_expired(
        self, connection: sqlite3.Connection, timestamp: str
    ) -> None:
        now = datetime.fromisoformat(timestamp)
        cutoff = (now - timedelta(days=self.inactivity_days)).isoformat()
        connection.execute(
            "DELETE FROM reminders WHERE last_referenced_at < ?", (cutoff,)
        )

    @classmethod
    def _clean_label(cls, value: str, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError(f"{name} must not be empty")
        if len(normalized) > 160:
            raise ValueError(f"{name} must be 160 characters or fewer")
        if _SENSITIVE_LABEL.search(normalized):
            raise ValueError(
                f"{name} must not contain credentials, account identifiers, or amounts"
            )
        return normalized

    @classmethod
    def _clean_keywords(cls, keywords: Iterable[str]) -> Tuple[str, ...]:
        if isinstance(keywords, str):
            raise TypeError("keywords must be a collection of phrases")
        clean = []
        seen = set()
        for keyword in keywords:
            normalized = cls._clean_label(keyword, "keyword")
            key = normalized.casefold()
            if key in seen:
                raise ValueError("keywords must be unique")
            seen.add(key)
            clean.append(normalized)
        if not clean:
            raise ValueError("at least one keyword is required")
        if len(clean) > 16:
            raise ValueError("at most 16 keywords may be stored per reminder")
        return tuple(clean)

    @staticmethod
    def _timestamp(now: Optional[datetime]) -> str:
        value = datetime.now(timezone.utc) if now is None else now
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _normalize_phrase(value: str) -> str:
        return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))

    @classmethod
    def _contains_phrase(cls, normalized_query: str, phrase: str) -> bool:
        normalized_phrase = cls._normalize_phrase(phrase)
        return bool(
            normalized_phrase
            and f" {normalized_phrase} " in f" {normalized_query} "
        )

    @staticmethod
    def _from_row(
        row: sqlite3.Row, last_referenced_at: Optional[str] = None
    ) -> Reminder:
        return Reminder(
            id=row["id"],
            topic=row["topic"],
            keywords=tuple(json.loads(row["keywords"])),
            due_on=date.fromisoformat(row["due_on"]) if row["due_on"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            last_referenced_at=datetime.fromisoformat(
                last_referenced_at or row["last_referenced_at"]
            ),
        )
