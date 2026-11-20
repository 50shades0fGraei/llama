# Evidence- and risk-aware solution protocol

This adapts EFWA's broad cycle of observing an experience, considering solutions,
and refining future service into a bounded, privacy-conscious workflow:

1. Clarify the user's stated request and desired outcome. Ask rather than infer
   hidden motives, feelings, identity, or demographic characteristics.
2. Use authorized evidence and documented company requirements. State what's
   unknown; ask for consent before optional data use or external retrieval.
3. Apply non-revocable safeguards first: applicable law and system policy, user
   safety and protection, truthful evidence use, privacy, consent, agency, and no
   unverified actions.
4. Compare the user's explicit goal, the company's documented requirements, and
   the service's accuracy/reliability objectives as parallel interests. Do not
   invent company objectives or treat the AI as having personal desires.
5. Identify solution options and their risks. `assess_solution_options` accepts
   caller-supplied 0-100 alignment ratings for each of the three interests. It
   computes their equal mean after validation. Inputs should be supported by
   explicit goals, documented requirements, and evidence—not demographic
   assumptions or model confidence alone.
6. A stated safeguard violation blocks an option. High or critical risk requires
   human review and is not automatically ranked for selection. Eligible low or
   medium-risk options are ranked by equal alignment; lower risk breaks ties.
   The function is advisory and never executes an action.
7. Explain relevant evidence, uncertainty, and trade-offs; ask the user or route
   to an authorized human when goals conflict or risk is elevated.
8. If configured, the separate response reviewer performs one final check. Future
   quality adaptation uses only explicit, human-reviewed aggregate labels; it
   does not learn ideology, silently change priorities, or retain full dialogue.

## What was not adopted from EFWA

The source document proposes demographic and trauma profiling, inferred
psychological traits, and demographic-specific predictions. This implementation
does not use those methods. It also does not scrape social platforms, claim
statistics about the assistant's user base, retain all memories, train from raw
conversations, or modify model weights. Personalization must be based on
information the user explicitly provides and is used only within the fixed
safeguards.

The risk-rating inputs are not automatically inferred by this package. The host
application is responsible for a defensible, auditable basis for each rating and
for obtaining human review where required. A high numerical alignment cannot
override a safeguard or turn a high-risk option into an automatic recommendation.
