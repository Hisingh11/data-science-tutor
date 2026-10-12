import unittest

from utils.self_rag_graph import SelfRAGEngine


class _EmptyRag:
    def relevant(self, query, n_results):
        return []


class _CalculationReviewModel:
    def __init__(self):
        self.answers = []
        self.reviews = 0
        self.review_contexts = []

    def answer(self, prompt, **kwargs):
        self.answers.append(prompt)
        if len(self.answers) == 1:
            return "Initial draft with an inconsistent total. " + ("detail " * 400) + "DRAFT_END_SENTINEL"
        return "Corrected calculation."

    def complete_json(self, prompt, schema, name, *args, **kwargs):
        self.reviews += 1
        self.review_prompt = prompt
        self.review_schema = schema
        self.review_contexts.append(kwargs["context"])
        if self.reviews == 1:
            return {
                "claims": [],
                "useful": "high",
                "missing": [],
                "calculation_errors": ["The closing balance does not reconcile with deposits and gain."],
            }
        return {
            "claims": [],
            "useful": "high",
            "missing": [],
            "calculation_errors": [],
        }


class SelfRAGCalculationTests(unittest.TestCase):
    def test_calculation_review_error_triggers_corrective_retry(self):
        model = _CalculationReviewModel()
        engine = SelfRAGEngine(_EmptyRag(), model)

        answer = engine.run(
            "Calculate the total investment and gain.",
            "Calculate the total investment and gain.",
        )

        self.assertEqual(answer, "Corrected calculation.")
        self.assertEqual(len(model.answers), 2)
        self.assertIn("Calculation errors:", model.answers[1])
        self.assertIn("Recalculate numeric results", model.answers[1])
        self.assertIn("DRAFT_END_SENTINEL", model.answers[1])
        self.assertIn("DRAFT_END_SENTINEL", model.review_contexts[0])
        self.assertIn("Complete draft answer", model.review_contexts[0])
        self.assertIn("calculation_errors", model.review_schema["required"])
        self.assertIn("independently audit", model.review_prompt)
        self.assertEqual(engine.last_log["retries_used"], 1)
        self.assertEqual(engine.last_log["calculation_errors"], [])


if __name__ == "__main__":
    unittest.main()
