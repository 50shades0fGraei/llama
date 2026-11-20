import unittest

from emotional_ai import (
    CustomerServiceAssistant,
    assess_customer_message,
    handoff_metrics,
)


class FakeGenerator:
    def __init__(self):
        self.dialogs = None
        self.options = None

    def chat_completion(
        self,
        dialogs,
        temperature=0.6,
        top_p=0.9,
        max_gen_len=None,
        logprobs=False,
    ):
        self.dialogs = dialogs
        self.options = (temperature, top_p, max_gen_len, logprobs)
        return [{"generation": {"role": "assistant", "content": "I can help."}}]


class DecisionTests(unittest.TestCase):
    def test_detects_only_explicit_support_cues(self):
        assessment = assess_customer_message(
            "I'm frustrated. Please connect me with a person."
        )

        self.assertTrue(assessment.acknowledgement_recommended)
        self.assertTrue(assessment.human_handoff_requested)
        self.assertTrue(assessment.human_handoff_recommended)
        self.assertEqual(
            assessment.signals,
            (
                "explicit_human_support_request",
                "explicitly_reported_distress",
            ),
        )

    def test_ignores_negated_emotion_and_human_request(self):
        assessment = assess_customer_message(
            "I am not frustrated and I don't want to speak with a human."
        )

        self.assertFalse(assessment.acknowledgement_recommended)
        self.assertFalse(assessment.human_handoff_requested)
        self.assertFalse(assessment.human_handoff_recommended)
        self.assertEqual(assessment.signals, ())

    def test_urgency_is_a_wording_cue_not_a_handoff_decision(self):
        assessment = assess_customer_message("There is an unauthorized charge.")

        self.assertTrue(assessment.urgent_language)
        self.assertFalse(assessment.human_handoff_requested)
        self.assertTrue(assessment.human_handoff_recommended)

    def test_rejects_non_string_messages(self):
        with self.assertRaises(TypeError):
            assess_customer_message(None)


class AssistantTests(unittest.TestCase):
    def test_adds_context_without_mutating_dialog_and_forwards_options(self):
        generator = FakeGenerator()
        assistant = CustomerServiceAssistant(generator)
        dialog = [
            {"role": "system", "content": "Follow the store policy."},
            {"role": "user", "content": "I'm upset. My order still isn't working."},
        ]
        original_dialog = [message.copy() for message in dialog]

        result = assistant.chat_completion(
            [dialog], temperature=0.2, top_p=0.8, max_gen_len=32, logprobs=True
        )

        self.assertEqual(dialog, original_dialog)
        self.assertEqual(generator.options, (0.2, 0.8, 32, True))
        self.assertEqual(result[0]["generation"]["content"], "I can help.")
        system_message = generator.dialogs[0][0]["content"]
        self.assertIn("Follow the store policy.", system_message)
        self.assertIn("Acknowledge the stated service problem", system_message)

    def test_requests_human_next_step_without_claiming_transfer(self):
        generator = FakeGenerator()
        assistant = CustomerServiceAssistant(generator)

        assistant.chat_completion(
            [[{"role": "user", "content": "I need to talk to an agent."}]]
        )

        system_message = generator.dialogs[0][0]["content"]
        self.assertIn("explicitly requested a person", system_message)
        self.assertIn("do not claim a transfer was completed", system_message)

    def test_rejects_invalid_dialogs(self):
        assistant = CustomerServiceAssistant(FakeGenerator())

        with self.assertRaises(ValueError):
            assistant.chat_completion([])
        with self.assertRaises(ValueError):
            assistant.chat_completion([[]])
        with self.assertRaises(ValueError):
            assistant.chat_completion(
                [[{"role": "assistant", "content": "Hello"}]]
            )

    def test_rejects_empty_system_prompt(self):
        with self.assertRaises(ValueError):
            CustomerServiceAssistant(FakeGenerator(), system_prompt=" ")


class MetricsTests(unittest.TestCase):
    def test_reports_handoff_precision_recall_and_f1(self):
        result = handoff_metrics(
            expected=[True, True, False, False],
            predicted=[True, False, True, False],
        )

        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 1)
        self.assertEqual(result.true_negatives, 1)
        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.recall, 0.5)
        self.assertEqual(result.f1, 0.5)

    def test_rejects_empty_or_mismatched_labels(self):
        with self.assertRaises(ValueError):
            handoff_metrics([], [])
        with self.assertRaises(ValueError):
            handoff_metrics([True], [])
        with self.assertRaises(TypeError):
            handoff_metrics([1], [True])


if __name__ == "__main__":
    unittest.main()
