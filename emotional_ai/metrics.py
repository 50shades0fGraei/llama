from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HandoffMetrics:
    examples: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def handoff_metrics(
    expected: Sequence[bool], predicted: Sequence[bool]
) -> HandoffMetrics:
    """Score human-handoff recommendations against a labeled evaluation set."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    if not expected:
        raise ValueError("at least one labeled example is required")
    if any(type(value) is not bool for value in expected) or any(
        type(value) is not bool for value in predicted
    ):
        raise TypeError("expected and predicted values must be bool")

    true_positives = sum(actual and guess for actual, guess in zip(expected, predicted))
    false_positives = sum(
        not actual and guess for actual, guess in zip(expected, predicted)
    )
    false_negatives = sum(
        actual and not guess for actual, guess in zip(expected, predicted)
    )
    true_negatives = sum(
        not actual and not guess for actual, guess in zip(expected, predicted)
    )

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    examples = len(expected)
    return HandoffMetrics(
        examples=examples,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        accuracy=(true_positives + true_negatives) / examples,
        precision=precision,
        recall=recall,
        f1=f1,
    )
