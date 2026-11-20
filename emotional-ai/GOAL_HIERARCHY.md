# Fixed and adaptive goal hierarchy

The assistant uses two distinct priority levels and a presentation layer:

1. **Fixed core hierarchy:** applicable law and system policy, plus user safety
   and protection, are non-overridable. Truthfulness, evidence-grounding,
   uncertainty, privacy, consent, user agency, and not claiming unverified
   actions are also fixed safeguards. These cannot be overridden by preferences,
   generated opinions, reviewer output, or feedback.
2. **Adaptive priorities:** a caller may provide up to eight explicit goals for a
   request, each with a weight from 1 to 100 and a source of `user_explicit`,
   `task`, or `human_review`. The assistant uses these only to order otherwise
   acceptable ways of helping. They are preferences—not evidence, policy, or
   permission to infer hidden interests.
3. **Parallel stakeholder interests:** when assessing a solution, consider the
   user's stated goals, the company's documented requirements, and the service's
   reliability/accuracy objectives as co-equal considerations beneath the fixed
   safeguards. `assess_solution_options` uses an equal mean of three caller-
   supplied alignment ratings; it does not infer a company's interests or the
   AI's personal desires.
4. **Presentation choices:** tone, detail, and efficiency are applied after the
   higher priorities are met.

The assistant does not generate its own goals, adopt an ideology, or learn new
priority weights from its own opinions. Human-reviewed quality feedback can add
bounded task guidance, but it does not change this hierarchy.

For `task="opinion"`, the assistant should separate sourced facts from judgment,
identify uncertainty, and compare relevant perspectives using evidence and
criteria the user or caller actually supplied. Adaptive weights can influence
which acceptable considerations are emphasized; they cannot make a weakly
supported claim more factual. If the evidence is insufficient, say so instead
of assigning false precision.

The solution/risk assessment is advisory. Explicit safeguard violations are
blocked; high or critical risks require human review and are not automatically
ranked for selection. Low and medium risk options are ranked by equal stakeholder
alignment, with lower risk used only to break ties. See
[`SOLUTION_PROTOCOL.md`](SOLUTION_PROTOCOL.md).

```python
from emotional_ai import CustomerServiceAssistant, GoalPriority

support = CustomerServiceAssistant(generator)
reply = support.chat_completion(
    [[{"role": "user", "content": "Compare these options."}]],
    task="opinion",
    goal_priorities=[
        GoalPriority("include alternatives", 80, "user_explicit"),
        GoalPriority("be concise", 40, "task"),
    ],
)
```

`CORE_GOAL_HIERARCHY` and `rank_goal_priorities` are also available from
`emotional_ai`. Priorities are request-scoped and are not persisted by the
framework.
