# Optional customer-service support layer

`emotional_ai` adds a small, opt-in support layer around an existing Llama-compatible
chat generator. It keeps the model's existing weights and adds system-message
guidance based on a few explicit wording cues. It does not fine-tune model weights,
diagnose a customer's emotional state, or perform account actions.

For consent-based ranking of preferences the customer explicitly provides, see
[`emotional-ai/INTEREST_HIERARCHY.md`](emotional-ai/INTEREST_HIERARCHY.md).

Assessment and support routing happen before the wrapped generator tokenizes the
augmented dialog. An optional `response_style` (`"neutral"`, `"warm"`, or
`"direct"`) is a caller-provided preference; it is not inferred from emotion.
Choose `task="general"`, `"troubleshooting"`, `"policy"`, `"recommendation"`, or
`"opinion"` explicitly when calling `CustomerServiceAssistant.chat_completion`.

Optional, separately configured persistence:

- `MemoryStore.from_environment()` returns no store unless
  `EMOTIONAL_AI_MEMORY_ENABLED=1`; an application may also construct a store
  explicitly. It stores consented reminder metadata in a profile-local SQLite
  file, matches saved keywords against future messages, deletes resolved items,
  and expires items after 90 days without a match. See
  [`emotional-ai/MEMORY_RETENTION.md`](emotional-ai/MEMORY_RETENTION.md).
- `QualityMemoryStore.from_environment()` returns no store unless
  `EMOTIONAL_AI_QUALITY_ENABLED=1`; an application may also construct a store
  explicitly. It stores response-quality labels without prompts or answers and
  only adds prompt hints from repeated human-reviewed feedback. See
  [`emotional-ai/QUALITY_MEMORY.md`](emotional-ai/QUALITY_MEMORY.md).
- `EvidenceRetriever` is optional and receives the current user message and task
  for each response. A remote retriever may disclose that query; use a local source
  or obtain user consent and minimize what is sent.
- `ModelResponseReviewer` optionally runs one separate review-model pass after a
  draft. It checks task fit, support from supplied evidence, uncertainty, tone,
  pressure, and consistent self-presentation, then returns only the final answer.
  It is not a continuous background process and does not create personal beliefs
  or ideology. See
  [`emotional-ai/RESPONSE_REVIEW.md`](emotional-ai/RESPONSE_REVIEW.md).
- Opinion tasks separate factual claims from value judgments and handle
  unresolved disagreements by surfacing trade-offs rather than forcing consensus.
  Anger alone does not imply crisis or trigger hotline language; explicit safety
  concerns remain eligible for direct support. See
  [`emotional-ai/BIAS_AND_TONE.md`](emotional-ai/BIAS_AND_TONE.md).
- `CORE_GOAL_HIERARCHY` is fixed. A caller may pass request-scoped weighted
  `GoalPriority` values to adjust emphasis within those constraints; model
  opinions cannot alter the hierarchy. See
  [`emotional-ai/GOAL_HIERARCHY.md`](emotional-ai/GOAL_HIERARCHY.md).
- `assess_solution_options` scores explicit user, company, and service alignment
  equally after hard safeguards. It blocks declared safeguard violations and
  routes high/critical risk options to human review; it never executes an action.
  See [`emotional-ai/SOLUTION_PROTOCOL.md`](emotional-ai/SOLUTION_PROTOCOL.md).
- `select_contextual_ad` is a separate, opt-in workflow that returns a
  clearly labeled sponsored card based on the current message's general topic.
  It does not target mood or demographics or modify assistant output. See
  [`emotional-ai/CHAT_ADS.md`](emotional-ai/CHAT_ADS.md).

The raw user message is not stored by the reminder or quality ledgers. These
SQLite files are local but not encrypted by this package; use a separate memory
path for each user profile.

```python
from emotional_ai import (
    CustomerServiceAssistant,
    GoalPriority,
    MemoryStore,
    ModelResponseReviewer,
    QualityMemoryStore,
)

support = CustomerServiceAssistant(
    generator,  # an existing Llama instance
    memory_store=MemoryStore.from_environment(),
    quality_store=QualityMemoryStore.from_environment(),
    reviewer=ModelResponseReviewer(review_model),  # optional separate model
)
reply = support.chat_completion(
    [[{"role": "user", "content": "Compare these options."}]],
    task="opinion",
    goal_priorities=[
        GoalPriority("include alternatives", 80, "user_explicit"),
        GoalPriority("be concise", 40, "task"),
    ],
)
```

To enable evidence search, pass an app-specific implementation of
`EvidenceRetriever`; its `search` method receives the current user message and
task. See [`emotional-ai/EVIDENCE_RETRIEVAL.md`](emotional-ai/EVIDENCE_RETRIEVAL.md).

Use `assess_solution_options` for structured option and risk review; its
stakeholder ratings must come from explicit goals and documented requirements,
not inferred demographic or psychological traits.

`assess_customer_message` returns transparent cues for explicitly reported
frustration, service-problem wording, requests for human support, and urgency-related
wording. A handoff is recommended only when a person is explicitly requested or
urgency-related wording is detected. These are keyword-based signals, not a
validated emotion classifier or automatic routing action. Review the signals and
any generated answer before using them in a customer-facing workflow.

## Evaluating service quality

`handoff_metrics` compares handoff recommendations against a human-labeled,
representative test set and reports precision, recall, F1, and accuracy. Track
false positives and false negatives separately; missing a needed handoff and
unnecessarily escalating a routine request have different costs.

For a real deployment, also measure customer-rated helpfulness, first-contact
resolution, time to resolution, and policy accuracy against a baseline using the
same test set and operating conditions. Do not claim an advantage over competing
systems without that comparison. Collect the minimum data needed, obtain
appropriate consent, and avoid demographic or psychological profiling.

## Authorship and contribution

The Emotional AI framework, original workflow concepts, and source materials are
authored by Randall Lujan. The repository implementation and safety-focused
conversation-guidance adaptation were developed collaboratively by Randall Lujan
with Copilot as a coding assistant.
