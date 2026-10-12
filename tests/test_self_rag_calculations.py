import unittest

from utils.self_rag_graph import SelfRAGEngine


class _EmptyRag:
    def relevant(self, query, n_results):
        return []


class _DirectAnswerModel:
    def __init__(self):
        self.answer_calls = []

    def answer(self, prompt, **kwargs):
        self.answer_calls.append((prompt, kwargs))
        return "The direct tutor answer."

    def complete_json(self, *args, **kwargs):
        raise AssertionError("Tutor responses must not make a self-check model call.")


class SelfRAGDirectAnswerTests(unittest.TestCase):
    def test_tutor_generates_once_without_reflection_or_retry(self):
        model = _DirectAnswerModel()
        engine = SelfRAGEngine(_EmptyRag(), model)

        answer = engine.run(
            "Calculate the total investment and gain.",
            "Calculate the total investment and gain.",
            history=[],
        )

        self.assertEqual(answer, "The direct tutor answer.")
        self.assertEqual(len(model.answer_calls), 1)
        self.assertEqual(model.answer_calls[0][1]["max_tokens"], 4096)
        self.assertEqual(model.answer_calls[0][1]["temperature"], 0.2)
        self.assertIn("Answer completely and at useful depth", model.answer_calls[0][0])
        self.assertEqual(engine.last_log["retrieved"], [])


if __name__ == "__main__":
    unittest.main()
