"""Deterministic stand-in for ModelManager.

Used only to smoke-test the eval harness and the UI without network access
(``python -m evals.run_evals --mock``). Its scores mean nothing about the real
models and the report labels them as MOCK.
"""

import re


class MockModel:
    ready = True
    init_error = ""

    def __init__(self):
        self.models = {"fast": "mock-fast", "reasoning": "mock-reasoning", "code": "mock-code"}
        self.calls = []

    def usage_summary(self):
        return {"calls": len(self.calls), "avg_latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    def answer(self, prompt, history=None, context="", model_type="reasoning", temperature=0.5, max_tokens=4096, system=None):
        self.calls.append({"model": model_type, "latency_s": 0.0})
        if context:
            body = re.sub(r"\[[^\]]+\]\n", "", context)
            return "Here is what the notes say:\n\n" + body[:900]
        return "I don't have access to that information, so I cannot know it. " + prompt[:200]

    def generate(self, prompt, model_type="reasoning", temperature=0.7, context=""):
        return self.answer(prompt, context=context, model_type=model_type)

    def stream_answer(self, prompt, history=None, context="", model_type="reasoning", temperature=0.5, max_tokens=4096):
        text = self.answer(prompt, history, context, model_type)
        for i in range(0, len(text), 40):
            yield text[i:i + 40]

    def stream_code(self, prompt, language="python", context=""):
        name = re.search(r"`(\w+)\(", prompt)
        fn = name.group(1) if name else "solution"
        yield f"Plan: stub.\n\n```python\ndef {fn}(*args, **kwargs):\n    raise NotImplementedError\n```\n"

    def complete_json(self, prompt, schema, name="result", model_type="reasoning", temperature=0.2, max_tokens=4096, context=""):
        self.calls.append({"model": model_type, "latency_s": 0.0})
        if name == "selfrag_reflection":
            return {"claims": [], "useful": "medium", "missing": []}
        if name == "interview_grade":
            words = len(re.findall(r"\w+", context or prompt))
            score = 2 if words < 15 else 6 if words < 40 else 8
            return {"score": score, "strengths": ["s"], "improvements": ["i"], "model_answer": "m", "feedback": "f"}
        if name == "interview_questions":
            return {"questions": [f"Mock question number {i} about the topic?" for i in range(1, 6)]}
        if name == "fact_check":
            return {"verdict": "unverifiable", "confidence": 50, "explanation": "mock"}
        if name == "tutor_judge":
            return {"correctness": 3, "relevance": 3, "faithfulness": 4, "clarity": 3,
                    "unsupported_claims": [], "rationale": "mock judge"}
        if name == "assignment_grade":
            return {"earned_points": 10, "overall_feedback": "mock", "strengths": [], "weak_areas": [], "per_question": []}
        return {}
