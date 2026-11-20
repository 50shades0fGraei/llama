# Quality feedback memory

Quality memory is separate from a user's reminder profile. It persists only
explicit labels about response quality, such as `too_verbose`, `incorrect`,
`policy_gap`, `tone_mismatch`, or `offensive_or_unsafe`. It stores the task,
category, severity, feedback source, and timestamp—not prompts, answers, or user
identifiers.

Enable it with `EMOTIONAL_AI_QUALITY_ENABLED=1`. Set
`EMOTIONAL_AI_QUALITY_PATH` to choose a local SQLite file; otherwise the default is
`~/.emotional_ai/quality.sqlite3`. It is disabled by default and is not encrypted
by this feature.

Call `record_feedback` only after an explicit user rating or human review. User
feedback is counted but does not directly tune future prompts. Prompt hints are
derived only from repeated human-reviewed labels (three by default); a
human-reviewed critical label also adds a caution. `guidance_for(task)` supplies
those hints to the next generation for that task. It does not update model weights
or rewrite safety policy.

```python
from emotional_ai import QualityMemoryStore

quality = QualityMemoryStore.from_environment()
if quality is not None:
    quality.record_feedback(
        task="opinion",
        category="too_verbose",
        severity="low",
        source="human_review",
    )
```

Use `quality.summary(task)` to inspect aggregate counts and `quality.clear()` to
delete the local feedback ledger. A human must review trends and decide whether
prompt or workflow changes are justified; the system does not learn from a single
mistake or silently change its objectives.
