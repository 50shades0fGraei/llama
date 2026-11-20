# Customer-support decision workflow

Use the following steps to guide a response. This is a support aid, not an
autonomous account-action or customer-profiling system.

1. **Check explicit communication cues.** Acknowledge only feelings the customer
   states and note literal urgency or requests for a person. Do not infer hidden
   emotions or predict a reaction.
2. **Parse the task.** Identify what the customer actually asked for. Do not replace
   that goal with an assumed goal.
3. **Check authority.** Use the documented product policy and available account
   data. If a fact or permission is missing, ask or say what cannot be verified.
4. **Choose a response path.**
   - Routine question: answer with a source-backed, concise explanation.
   - Unclear request: ask one neutral clarifying question.
   - Explicit request for a person: offer the real human-support route.
   - Explicit urgent or high-impact wording: prioritize clarification and offer
     timely human review. Treat wording as a cue, not proof of an emergency.
5. **Apply the user's chosen response style.** Use it only when the user explicitly
   selected a tone such as warm, neutral, or direct. Do not choose tone to exploit
   emotion or produce a predicted reaction.
6. **Apply the caller-selected task workflow.** Choose `general`,
   `troubleshooting`, `policy`, or `recommendation` explicitly. The workflow adds
   task-specific guidance; it does not infer a hidden objective from the user.
7. **Preserve agency.** Present options and relevant trade-offs without pressure.
   Let the customer reject suggestions or change direction.
8. **Confirm completion.** Claim an action only after the connected system confirms
   it. Otherwise describe the next step as a suggestion, not a completed transfer
   or transaction.

The optional `emotional_ai` Python layer assesses explicit wording first, then
selects a support route before handing the augmented dialog to Llama for
tokenization and generation. Its recommendations must not be used as the sole
basis for denying service, making a consequential decision, or taking an account
action. A person should review edge cases and escalation rules before deployment.
