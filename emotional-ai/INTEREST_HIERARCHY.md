# Opt-in interest hierarchy

An interest hierarchy is a short-lived ordering of topics the customer explicitly
chooses to help tailor a relevant answer. It is not a profile to infer from
emotional cues, demographics, browsing history, or unrelated conversation.

## Conversation flow

1. Explain why preferences would help with the current request.
2. Ask permission before collecting them.
3. If the customer agrees, invite them to name and order the relevant topics.
4. Repeat the order back and invite corrections.
5. Use the order only for the stated purpose. Do not retain it unless the customer
   separately opts in to a disclosed save feature.
6. Make it easy to skip, revise, or clear the preferences.

Example: “Would you like to rank the topics you want me to prioritize for these
recommendations? You can skip or change the list.”

## Python API

`rank_user_stated_interests` accepts an already-consented list in the user's chosen
order. It only trims surrounding whitespace and assigns ranks; it does not inspect
conversation history, discover interests, or write data to storage.

```python
from emotional_ai import rank_user_stated_interests

hierarchy = rank_user_stated_interests(
    ["delivery speed", "price", "sustainability"],
    consented=True,
)
```

The caller must obtain genuine consent before passing preferences. Do not use the
result to exploit vulnerabilities, apply pressure, or make consequential decisions.
