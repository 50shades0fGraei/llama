# Customer-support decision workflow

Use the following steps to guide a response. This is a support aid, not an
autonomous account-action or customer-profiling system.

1. **Parse the request.** Restate the task in plain terms and use only details
   supplied for this request or explicitly saved with consent.
2. **Check authority.** Use the documented product policy and available account
   data. If a fact or permission is missing, ask or say what cannot be verified.
3. **Choose a response path.**
   - Routine question: answer with a source-backed, concise explanation.
   - Unclear request: ask one neutral clarifying question.
   - Explicit request for a person: offer the real human-support route.
   - Explicit urgent or high-impact wording: prioritize clarification and offer
     timely human review. Treat wording as a cue, not proof of an emergency.
4. **Preserve agency.** Present options and relevant trade-offs without pressure.
   Let the customer reject suggestions or change direction.
5. **Confirm completion.** Claim an action only after the connected system confirms
   it. Otherwise describe the next step as a suggestion, not a completed transfer
   or transaction.

The optional `emotional_ai` Python layer detects a small set of explicit wording
cues. Its recommendations must not be used as the sole basis for denying service,
making a consequential decision, or taking an account action. A person should
review edge cases and escalation rules before deployment.
