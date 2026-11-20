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

The wrapper evaluates only the final `user` message in each dialog; the normal
Llama `system`/`user`/`assistant` role structure remains intact. `task` is selected
by the caller and applies to every dialog in that call:

| Task | Guidance |
| --- | --- |
| `general` | Answer the stated request. This is the default. |
| `troubleshooting` | Offer safe, reversible steps one at a time. |
| `policy` | Use supplied policy and state what is unknown. |
| `recommendation` | Use stated criteria, ask consent for optional preferences, and show trade-offs. |

These are four curated workflows, not a dynamic loader for the archive's 33
documents. The archive documents are not executable modules.
