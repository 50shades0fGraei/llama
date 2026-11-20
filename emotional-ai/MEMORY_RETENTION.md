# Local reminder memory

Reminder memory is optional, local to an OS user/profile, and disabled unless the
application explicitly enables and passes a `MemoryStore`. It keeps only a short
topic, user-selected keywords, an optional date, and timestamps. It does not keep
the triggering sentence, full conversation, user ID, or tokenizer IDs.

## Enable and scope the store

Set `EMOTIONAL_AI_MEMORY_ENABLED=1` to enable the factory. Set
`EMOTIONAL_AI_MEMORY_PATH` to choose a different SQLite file; otherwise the file is
`~/.emotional_ai/memory.sqlite3`. Environment variables configure the store; they
are not used to hold memories. SQLite is not encrypted by this feature, and the OS
account running the process can read the file. Use a separate path per user profile
in multi-user applications.

```python
from emotional_ai import CustomerServiceAssistant, MemoryStore

memory = MemoryStore.from_environment()  # None unless explicitly enabled
support = CustomerServiceAssistant(generator, memory_store=memory)
```

## Save, recall, resolve

The application should identify a concrete future or unresolved item, show the
compact topic/date it proposes to save, and pass `consented=True` only after the
user opts in. Store no payment credentials, amounts, account identifiers, or raw
message text. Natural-language extraction is the host application's
responsibility; review the proposed topic, keywords, and date with the user before
calling the storage API.

```python
from datetime import date
from emotional_ai import MemoryStore

memory = MemoryStore.from_environment()
if memory is not None:
    reminder = memory.add_reminder(
        "utility payment",
        ["utility", "payment"],
        due_on=date(2027, 1, 15),
        consented=True,  # the application has recorded the user's opt-in
    )
```

When enabled, the assistant checks each new user message for exact matches to saved
topics/keywords before generation. A match refreshes the 90-day inactivity period
and may be included as JSON context. The assistant does not infer a date or save a
new reminder by itself; the application must create it after user consent.

Resolved reminders are deleted immediately with `memory.resolve(reminder.id)`.
Unresolved reminders expire after 90 days without a keyword match. `memory.clear()`
deletes the profile's entire reminder store. This module does not run a background
notification service; reminders are surfaced when a later message matches.
