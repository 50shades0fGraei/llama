# Bounded response review

`ModelResponseReviewer` is an optional second pass, configured with a
caller-supplied chat generator. After the main assistant drafts each answer, the
reviewer receives the prepared conversation, selected task, and draft. It checks
whether the answer addresses the request, fits the task and selected style, is
supported by supplied evidence, states uncertainty, avoids unsupported actions,
and is respectful and non-coercive. It also checks the solution/risk protocol:
after fixed safeguards, consider explicit user goals, documented company
requirements, and service reliability as parallel interests. It does not infer
unstated goals or approve high/critical risk options.

The reviewer runs once per generated response and only when explicitly passed to
`CustomerServiceAssistant(reviewer=...)`. It is not a continuously running
background process. It adds a model call and corresponding latency and compute;
the application can omit it when that cost is not appropriate. A distinct local
model can be used for review. If the review model returns malformed output, the
wrapper raises an error rather than silently treating an unreviewed draft as
reviewed.

The reviewer returns only a final response, not a rationale, score, or private
reasoning. Its role check means consistent, honest service behavior; it does not
give the AI consciousness, personal opinions, interests, or an evolving ideology.
For opinion tasks, it checks that sourced facts, judgment, and uncertainty are not
conflated. Human review remains necessary for consequential decisions and
important claims.

```python
from emotional_ai import CustomerServiceAssistant, ModelResponseReviewer

support = CustomerServiceAssistant(
    generator,
    reviewer=ModelResponseReviewer(review_model),
)
```

Both `generator` and `review_model` implement the chat-completion interface. If
the reviewer is a remote service, the conversation and draft are sent to it; use a
local reviewer or obtain any required consent and minimize sensitive context.
