# Evaluation and service metrics

Evaluate the support experience on a consented, representative set of examples
before deployment. Keep evaluation data separate from training and report the
sample size and limitations.

## Triage metrics

Use [`handoff_metrics`](../emotional_ai/metrics.py) to compare handoff
recommendations with human-labeled expectations:

- **Precision:** how often a recommended handoff was appropriate.
- **Recall:** how often cases needing a handoff were caught.
- **F1:** a combined view of precision and recall.
- **Accuracy:** overall agreement, reported alongside precision and recall.

Review false positives and false negatives separately; their costs differ.

## Service outcomes

Track customer-rated helpfulness, first-contact resolution, time to resolution,
policy accuracy, and repeat-contact rate. Compare against a baseline using the same
scenarios, operating conditions, and scoring rubric. Segment only where lawful,
necessary, and consented; do not infer sensitive traits to improve a metric.

Do not claim to outperform competitors without a reproducible head-to-head
evaluation. A metric is not a target to optimize at the expense of customer choice,
privacy, or accurate answers.
