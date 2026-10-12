import unittest
from unittest.mock import patch

from utils.self_rag_graph import SelfRAGEngine


class _EmptyRag:
    def __init__(self):
        self.web_search_calls = []

    def relevant(self, query, n_results):
        return []

    def web_notes(self, query, max_results):
        self.web_search_calls.append((query, max_results))
        return [{
            "title": "Latest pandas release",
            "body": "The latest pandas release includes new features.",
            "href": "https://example.com/pandas",
        }]

    @staticmethod
    def _anchored(query, text):
        ignored = {"search", "online", "for", "the", "what", "is"}
        query_terms = set(query.lower().split()) - ignored
        text_terms = set(text.lower().split())
        return bool(query_terms & text_terms)


class _DirectAnswerModel:
    def __init__(self):
        self.answer_calls = []
        self.stream_calls = []

    def answer(self, prompt, **kwargs):
        self.answer_calls.append((prompt, kwargs))
        return "The direct tutor answer."

    def stream_answer(self, prompt, **kwargs):
        self.stream_calls.append((prompt, kwargs))
        yield "The direct "
        yield "tutor answer."

    def complete_json(self, *args, **kwargs):
        raise AssertionError("Tutor responses must not make a self-check model call.")


class SelfRAGDirectAnswerTests(unittest.TestCase):
    def test_tutor_generates_once_without_reflection_or_retry(self):
        rag = _EmptyRag()
        model = _DirectAnswerModel()
        engine = SelfRAGEngine(rag, model)

        answer = engine.run(
            "Calculate the total investment and gain.",
            "Calculate the total investment and gain.",
            history=[],
        )

        self.assertEqual(answer, "The direct tutor answer.")
        self.assertEqual(model.answer_calls, [])
        self.assertEqual(len(model.stream_calls), 1)
        self.assertEqual(model.stream_calls[0][1]["max_tokens"], 4096)
        self.assertEqual(model.stream_calls[0][1]["temperature"], 0.2)
        self.assertIn("Answer completely and at useful depth", model.stream_calls[0][0])
        self.assertEqual(engine.last_log["retrieved"], [])
        self.assertEqual(rag.web_search_calls, [])

    def test_explicit_online_request_uses_duckduckgo_without_extra_llm_call(self):
        rag = _EmptyRag()
        model = _DirectAnswerModel()
        engine = SelfRAGEngine(rag, model)

        with patch("utils.self_rag_graph.ENABLE_WEB_SEARCH", True):
            answer = engine.run(
                "Search online for the latest pandas release.",
                "Search online for the latest pandas release.",
                history=[],
            )

        self.assertEqual(answer, "The direct tutor answer.")
        self.assertEqual(
            rag.web_search_calls,
            [("Search online for the latest pandas release.", 5)],
        )
        self.assertEqual(model.answer_calls, [])
        self.assertEqual(len(model.stream_calls), 1)
        self.assertIn(
            "https://example.com/pandas", model.stream_calls[0][1]["context"]
        )
        self.assertEqual(
            engine.last_log["retrieved"],
            [{
                "topic": "Latest pandas release",
                "source": "web",
                "href": "https://example.com/pandas",
            }],
        )
        self.assertIn("cite provided URLs", model.stream_calls[0][1]["context"])

    def test_time_sensitive_question_uses_duckduckgo(self):
        rag = _EmptyRag()
        model = _DirectAnswerModel()
        engine = SelfRAGEngine(rag, model)

        with patch("utils.self_rag_graph.ENABLE_WEB_SEARCH", True):
            engine.run(
                "What is the latest pandas version?",
                "What is the latest pandas version?",
                history=[],
            )

        self.assertEqual(len(rag.web_search_calls), 1)


if __name__ == "__main__":
    unittest.main()
