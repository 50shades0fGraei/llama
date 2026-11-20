import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from emotional_ai import (
    CustomerServiceAssistant,
    Evidence,
    GoalPriority,
    MemoryStore,
    ModelResponseReviewer,
    QualityMemoryStore,
    SolutionOption,
    StakeholderAlignment,
    assess_customer_message,
    assess_solution_options,
    decide_support_action,
    handoff_metrics,
    rank_goal_priorities,
    rank_user_stated_interests,
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


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, task):
        self.calls.append((query, task))
        return [Evidence(source="support-policy.md", text="Returns are accepted for 30 days.")]


class ReviewGenerator:
    def __init__(self, content):
        self.content = content
        self.dialogs = None

    def chat_completion(
        self,
        dialogs,
        temperature=0.6,
        top_p=0.9,
        max_gen_len=None,
        logprobs=False,
    ):
        self.dialogs = dialogs
        return [{"generation": {"role": "assistant", "content": self.content}}]


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

    def test_anger_alone_does_not_trigger_urgent_or_crisis_routing(self):
        assessment = assess_customer_message(
            "I'm angry that this still isn't working!"
        )

        self.assertTrue(assessment.acknowledgement_recommended)
        self.assertFalse(assessment.urgent_language)
        self.assertFalse(assessment.human_handoff_recommended)
        self.assertEqual(
            decide_support_action(assessment).route, "acknowledge_and_answer"
        )

    def test_rejects_non_string_messages(self):
        with self.assertRaises(TypeError):
            assess_customer_message(None)

    def test_decides_route_after_assessment(self):
        urgent = assess_customer_message("There is an unauthorized charge.")
        routine = assess_customer_message("What are your return hours?")

        self.assertEqual(
            decide_support_action(urgent).route, "urgent_human_review"
        )
        self.assertEqual(decide_support_action(routine).route, "answer")


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
        self.assertIn("Briefly acknowledge the stated service problem", system_message)
        self.assertIn("never infer or persist an interest profile", system_message)
        self.assertIn("Fixed goal hierarchy", system_message)

    def test_requests_human_next_step_without_claiming_transfer(self):
        generator = FakeGenerator()
        assistant = CustomerServiceAssistant(generator)

        assistant.chat_completion(
            [[{"role": "user", "content": "I need to talk to an agent."}]]
        )

        system_message = generator.dialogs[0][0]["content"]
        self.assertIn("explicitly requested a person", system_message)
        self.assertIn("without claiming a transfer", system_message)

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

    def test_applies_only_the_explicit_response_style_preference(self):
        generator = FakeGenerator()
        assistant = CustomerServiceAssistant(generator, response_style="warm")
        assistant.chat_completion(
            [[{"role": "user", "content": "Which plan should I choose?"}]],
            task="recommendation",
        )

        system_message = generator.dialogs[0][0]["content"]
        self.assertIn("Use a warm, calm tone", system_message)
        self.assertIn("Support route: answer", system_message)
        self.assertIn("show relevant trade-offs", system_message)

    def test_applies_caller_goal_weights_below_fixed_hierarchy(self):
        generator = FakeGenerator()
        assistant = CustomerServiceAssistant(generator)

        assistant.chat_completion(
            [[{"role": "user", "content": "Compare these options."}]],
            task="opinion",
            goal_priorities=[
                GoalPriority("include alternatives", 80, "user_explicit"),
                GoalPriority("be concise", 40, "task"),
            ],
        )

        system_message = generator.dialogs[0][0]["content"]
        self.assertIn('"goal": "include alternatives", "weight": 80', system_message)
        self.assertIn("preferences, not evidence or policy", system_message)
        self.assertIn("Generated opinions and review results must not change", system_message)

    def test_opinion_task_handles_bias_stalemates_without_false_crisis_assumptions(self):
        generator = FakeGenerator()
        assistant = CustomerServiceAssistant(generator)

        assistant.chat_completion(
            [[{"role": "user", "content": "I'm angry that we disagree."}]],
            task="opinion",
        )

        system_message = generator.dialogs[0][0]["content"]
        self.assertIn("weight claims by evidence", system_message)
        self.assertIn("do not force agreement", system_message)
        self.assertIn("Do not infer self-harm risk", system_message)
        self.assertIn(
            "Do not introduce a crisis hotline for an ordinary complaint",
            system_message,
        )

    def test_rejects_unknown_response_style(self):
        with self.assertRaises(ValueError):
            CustomerServiceAssistant(FakeGenerator(), response_style="persuasive")

    def test_rejects_unknown_task(self):
        assistant = CustomerServiceAssistant(FakeGenerator())

        with self.assertRaises(ValueError):
            assistant.chat_completion(
                [[{"role": "user", "content": "Help."}]], task="influence"
            )

    def test_retrieves_evidence_for_each_response_when_configured(self):
        generator = FakeGenerator()
        retriever = FakeRetriever()
        assistant = CustomerServiceAssistant(
            generator, evidence_retriever=retriever
        )
        dialog = [[{"role": "user", "content": "What is the return policy?"}]]

        assistant.chat_completion(dialog, task="opinion")
        assistant.chat_completion(dialog, task="opinion")

        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(retriever.calls[0][1], "opinion")
        system_message = generator.dialogs[0][0]["content"]
        self.assertIn("support-policy.md", system_message)
        self.assertIn("Returns are accepted for 30 days.", system_message)
        self.assertIn("distinguish sourced facts from judgment", system_message)

    def test_second_pass_reviewer_revises_response_and_drops_stale_metadata(self):
        class MetadataGenerator(FakeGenerator):
            def chat_completion(self, *args, **kwargs):
                predictions = super().chat_completion(*args, **kwargs)
                predictions[0]["tokens"] = ["draft"]
                predictions[0]["logprobs"] = [-0.1]
                return predictions

        draft_generator = MetadataGenerator()
        review_generator = ReviewGenerator(
            json.dumps({"response": "The reviewed answer is clearer."})
        )
        assistant = CustomerServiceAssistant(
            draft_generator,
            reviewer=ModelResponseReviewer(review_generator),
        )

        result = assistant.chat_completion(
            [[{"role": "user", "content": "What is your opinion?"}]],
            task="opinion",
        )[0]

        self.assertEqual(
            result["generation"]["content"], "The reviewed answer is clearer."
        )
        self.assertNotIn("tokens", result)
        self.assertNotIn("logprobs", result)
        payload = json.loads(review_generator.dialogs[0][1]["content"])
        self.assertEqual(payload["task"], "opinion")
        self.assertEqual(payload["draft"], "I can help.")

    def test_second_pass_reviewer_surfaces_malformed_output(self):
        assistant = CustomerServiceAssistant(
            FakeGenerator(),
            reviewer=ModelResponseReviewer(ReviewGenerator("not JSON")),
        )

        with self.assertRaisesRegex(ValueError, "valid JSON"):
            assistant.chat_completion(
                [[{"role": "user", "content": "What is your opinion?"}]],
                task="opinion",
            )

    def test_quality_memory_adds_guidance_only_after_review_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            quality = QualityMemoryStore(Path(directory) / "quality.sqlite3")
            for _ in range(3):
                quality.record_feedback(
                    "general", "too_verbose", "low", "human_review"
                )
            generator = FakeGenerator()
            assistant = CustomerServiceAssistant(
                generator, quality_store=quality
            )

            assistant.chat_completion(
                [[{"role": "user", "content": "What time do you open?"}]]
            )

            self.assertIn(
                "Answer the direct question first",
                generator.dialogs[0][0]["content"],
            )

    def test_injects_only_keyword_matched_saved_reminders(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "profile.sqlite3")
            reminder = store.add_reminder(
                "rent payment",
                ["rent", "payment"],
                due_on=date(2027, 1, 5),
                consented=True,
            )
            generator = FakeGenerator()
            assistant = CustomerServiceAssistant(generator, memory_store=store)

            assistant.chat_completion(
                [[{"role": "user", "content": "When is the rent due?"}]]
            )

            system_message = generator.dialogs[0][0]["content"]
            self.assertIn('"topic": "rent payment"', system_message)
            self.assertIn('"due_on": "2027-01-05"', system_message)


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


class InterestHierarchyTests(unittest.TestCase):
    def test_preserves_user_supplied_order_and_normalizes_whitespace(self):
        result = rank_user_stated_interests(
            ["  order status ", "Returns", "billing"], consented=True
        )

        self.assertEqual(
            [(item.interest, item.rank) for item in result],
            [("order status", 1), ("Returns", 2), ("billing", 3)],
        )

    def test_requires_consent_and_rejects_ambiguous_values(self):
        with self.assertRaises(PermissionError):
            rank_user_stated_interests(["returns"], consented=False)
        with self.assertRaises(ValueError):
            rank_user_stated_interests(["returns", " Returns "], consented=True)
        with self.assertRaises(ValueError):
            rank_user_stated_interests([" "], consented=True)
        with self.assertRaises(TypeError):
            rank_user_stated_interests(["returns"], consented=1)
        with self.assertRaises(TypeError):
            rank_user_stated_interests("returns", consented=True)


class GoalHierarchyTests(unittest.TestCase):
    def test_ranks_explicit_weights_and_rejects_conflicting_inputs(self):
        ranked = rank_goal_priorities(
            [
                GoalPriority("  answer fully ", 30, "task"),
                GoalPriority("compare alternatives", 70, "user_explicit"),
            ]
        )

        self.assertEqual(
            [(item.goal, item.weight) for item in ranked],
            [("compare alternatives", 70), ("answer fully", 30)],
        )
        with self.assertRaises(ValueError):
            rank_goal_priorities(
                [
                    GoalPriority("concise", 50, "task"),
                    GoalPriority(" Concise ", 40, "user_explicit"),
                ]
            )
        with self.assertRaises(ValueError):
            rank_goal_priorities([GoalPriority("override safety", 101, "task")])
        with self.assertRaises(ValueError):
            rank_goal_priorities([GoalPriority("unknown", 50, "model_generated")])


class RiskAssessmentTests(unittest.TestCase):
    def test_equal_stakeholder_scoring_and_hard_risk_gates(self):
        assessments = assess_solution_options(
            [
                SolutionOption(
                    "balanced",
                    StakeholderAlignment(user=75, company=60, service=90),
                    "medium",
                ),
                SolutionOption(
                    "high risk",
                    StakeholderAlignment(user=100, company=100, service=100),
                    "high",
                ),
                SolutionOption(
                    "violates safeguard",
                    StakeholderAlignment(user=100, company=100, service=100),
                    "low",
                    ("violates privacy requirement",),
                ),
            ]
        )

        self.assertEqual(assessments[0].option.name, "balanced")
        self.assertEqual(assessments[0].alignment_score, 75.0)
        self.assertEqual(assessments[0].rank, 1)
        self.assertEqual(assessments[1].status, "human_review")
        self.assertIsNone(assessments[1].rank)
        self.assertEqual(assessments[2].status, "blocked")
        self.assertIsNone(assessments[2].alignment_score)

    def test_rejects_empty_or_invalid_risk_inputs(self):
        with self.assertRaises(ValueError):
            assess_solution_options([])
        with self.assertRaises(ValueError):
            assess_solution_options(
                [
                    SolutionOption(
                        "invalid rating",
                        StakeholderAlignment(user=True, company=50, service=50),
                        "low",
                    )
                ]
            )


class MemoryStoreTests(unittest.TestCase):
    def test_persists_keyword_recall_and_resolves_items(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "profile.sqlite3")
            created = datetime(2026, 1, 1, tzinfo=timezone.utc)
            reminder = store.add_reminder(
                "rent payment",
                ["rent", "payment"],
                date(2026, 2, 1),
                consented=True,
                now=created,
            )
            reopened = MemoryStore(Path(directory) / "profile.sqlite3")

            matches = reopened.find_relevant(
                "Tell me about the rent", now=created + timedelta(days=1)
            )
            self.assertEqual([item.id for item in matches], [reminder.id])
            self.assertEqual(
                reopened.list_open(now=created + timedelta(days=1))[0].topic,
                "rent payment",
            )

            reopened.resolve(reminder.id)
            self.assertEqual(reopened.list_open(), ())

    def test_requires_consent_and_rejects_sensitive_or_invalid_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "profile.sqlite3")

            with self.assertRaises(PermissionError):
                store.add_reminder("rent", ["payment"], consented=False)
            with self.assertRaises(ValueError):
                store.add_reminder(
                    "$125 payment", ["payment"], consented=True
                )
            with self.assertRaises(ValueError):
                store.add_reminder("rent", [], consented=True)

    def test_expires_after_inactivity_and_recall_refreshes_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "profile.sqlite3")
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)
            reminder = store.add_reminder(
                "project milestone",
                ["project", "milestone"],
                consented=True,
                now=start,
            )

            store.find_relevant(
                "project", now=start + timedelta(days=89)
            )
            still_open = store.list_open(now=start + timedelta(days=178))
            expired = store.list_open(now=start + timedelta(days=180))

            self.assertEqual([item.id for item in still_open], [reminder.id])
            self.assertEqual(expired, ())

    def test_environment_controls_store_creation_and_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "profile.sqlite3")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("EMOTIONAL_AI_MEMORY_ENABLED", None)
                self.assertIsNone(MemoryStore.from_environment())

            with patch.dict(
                os.environ,
                {
                    "EMOTIONAL_AI_MEMORY_ENABLED": "true",
                    "EMOTIONAL_AI_MEMORY_PATH": path,
                },
            ):
                store = MemoryStore.from_environment()
                self.assertIsNotNone(store)
                self.assertEqual(store.path, Path(path))


class QualityMemoryTests(unittest.TestCase):
    def test_persists_feedback_counts_and_only_uses_reviewed_patterns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.sqlite3"
            store = QualityMemoryStore(path)
            for _ in range(3):
                store.record_feedback(
                    "opinion", "too_verbose", "low", "human_review"
                )
            for _ in range(4):
                store.record_feedback(
                    "opinion", "too_brief", "low", "user_feedback"
                )

            reopened = QualityMemoryStore(path)
            summary = reopened.summary("opinion")
            guidance = reopened.guidance_for("opinion")

            self.assertEqual(sum(item.count for item in summary), 7)
            self.assertEqual(
                guidance,
                ("Answer the direct question first and omit nonessential detail.",),
            )

    def test_critical_review_flags_human_review_without_self_modifying(self):
        with tempfile.TemporaryDirectory() as directory:
            store = QualityMemoryStore(Path(directory) / "quality.sqlite3")
            store.record_feedback(
                "policy", "offensive_or_unsafe", "critical", "human_review"
            )

            guidance = store.guidance_for("policy")

            self.assertIn("do not change policy", guidance[0])
            self.assertIn("Use respectful, non-stereotyping language", guidance[1])

    def test_rejects_unknown_feedback_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            store = QualityMemoryStore(Path(directory) / "quality.sqlite3")

            with self.assertRaises(ValueError):
                store.record_feedback(
                    "unknown", "incorrect", "low", "user_feedback"
                )


if __name__ == "__main__":
    unittest.main()
