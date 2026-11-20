# Optional customer-service support layer

`emotional_ai` adds a small, opt-in support layer around an existing Llama-compatible
chat generator. It keeps the model's existing weights and adds system-message
guidance based on a few explicit wording cues. It does not fine-tune model weights,
diagnose a customer's emotional state, or perform account actions.

```python
from emotional_ai import CustomerServiceAssistant

support = CustomerServiceAssistant(generator)  # an existing Llama instance
reply = support.chat_completion(
    [[{"role": "user", "content": "I need to speak with a person."}]]
)
```

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
