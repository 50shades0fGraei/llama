# Emotional AI guidance

This folder turns the Emotional AI archive's relevant ideas into practical,
user-respecting conversation guidance. It is intended to complement the optional
Python wrapper documented in [`EMOTIONAL_AI.md`](../EMOTIONAL_AI.md).

## Files

- [`CONVERSATION_GUIDANCE.md`](CONVERSATION_GUIDANCE.md): how to acknowledge what
  customers say while preserving their agency.
- [`DECISION_WORKFLOW.md`](DECISION_WORKFLOW.md): a transparent support triage
  process and limits on automated decisions.
- [`INTEREST_HIERARCHY.md`](INTEREST_HIERARCHY.md): an opt-in way to rank
  preferences explicitly supplied by the customer.
- [`MEMORY_RETENTION.md`](MEMORY_RETENTION.md): profile-scoped, consent-based
  reminder storage and keyword recall.
- [`QUALITY_MEMORY.md`](QUALITY_MEMORY.md): a separate, local feedback ledger for
  reviewed quality trends.
- [`EVIDENCE_RETRIEVAL.md`](EVIDENCE_RETRIEVAL.md): optional source retrieval and
  privacy boundaries.
- [`GOAL_HIERARCHY.md`](GOAL_HIERARCHY.md): fixed constraints and caller-weighted
  adaptive priorities.
- [`SOLUTION_PROTOCOL.md`](SOLUTION_PROTOCOL.md): an evidence- and risk-aware
  solution review across parallel stakeholder goals.
- [`RESPONSE_REVIEW.md`](RESPONSE_REVIEW.md): an optional, bounded second-pass
  quality review.
- [`EVALUATION.md`](EVALUATION.md): metrics and a fair comparison plan.

These are curated rewrites, not copies of the archive. They replace demographic
profiling, psychological speculation, and covert persuasion with explicit,
consent-respecting conversation practices. The raw archive is not included: it
contains profiling and manipulation instructions, an account-export document,
trading code with credential fields, and large unreviewed documents.

At runtime, the wrapper assesses explicit wording cues, chooses a support route,
then applies a caller-selected task workflow and optional user-selected response
style, and passes the augmented dialog to the existing Llama generator. The
tokenizer and model weights are unchanged.

Optional reminder and quality stores can add per-profile memory and task-specific
reviewed guidance. Optional evidence retrieval can be configured for each response;
it is disabled by default.
An optional reviewer can perform one additional model pass before returning an
answer; it is disabled unless configured.
Caller-supplied goal weights can adjust emphasis for an individual request, but
cannot override the fixed safety, truthfulness, privacy, and agency constraints.

The wrapper evaluates only the final `user` message in each dialog; the normal
Llama `system`/`user`/`assistant` role structure remains intact. `task` is selected
by the caller and applies to every dialog in that call:

| Task | Guidance |
| --- | --- |
| `general` | Answer the stated request. This is the default. |
| `troubleshooting` | Offer safe, reversible steps one at a time. |
| `policy` | Use supplied policy and state what is unknown. |
| `recommendation` | Use stated criteria, ask consent for optional preferences, and show trade-offs. |
| `opinion` | Separate sourced facts from judgment and describe uncertainty. |

These are five curated workflows, not a dynamic loader for the archive's 33
documents. The archive documents are not executable modules.

The framework and workflow concepts are authored by Randall Lujan; the repository
implementation and guidance adaptation were developed with Copilot. See
[`EMOTIONAL_AI.md`](../EMOTIONAL_AI.md) for the contribution note.
