"""Eval suites for Data Scientist BOT.

Offline suites (no API key, deterministic):
  routing      – intent router accuracy + per-intent confusion
  retrieval    – BM25 Hit@1 / Hit@3 / MRR and off-topic rejection
  json_parsing – robustness of the JSON parser / schema sanitiser

Live suites (call Groq; cost tokens):
  tutor_qa          – Self-RAG answers: key-point recall + LLM-as-judge rubric,
                      prompt-injection and abstention checks, latency
  interview_grading – grader calibration against labelled score bands,
                      injection resistance
  code_gen          – pass@1: generated code is executed against unit tests
  fact_check        – verdict accuracy on labelled claims (needs web search)
"""

import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from typing import Callable, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "datasets")


def load(name: str, limit: int = 0) -> List[Dict]:
    with open(os.path.join(DATA, name), encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit else rows


def _pct(x: float) -> float:
    return round(100 * x, 1)


# ============================================================ offline suites
def suite_routing(limit=0, **_):
    from utils.router import classify_intent

    rows = load("routing.jsonl", limit)
    cases, per = [], defaultdict(lambda: [0, 0])
    confusion = Counter()
    for row in rows:
        got = classify_intent(row["text"])
        ok = got == row["expected"]
        per[row["expected"]][0] += ok
        per[row["expected"]][1] += 1
        if not ok:
            confusion[f"{row['expected']} → {got}"] += 1
        cases.append({"id": row["id"], "input": row["text"], "expected": row["expected"], "got": got, "pass": ok})
    acc = sum(c["pass"] for c in cases) / len(cases)
    return {
        "metrics": {"accuracy": _pct(acc), **{f"acc_{k}": _pct(v[0] / v[1]) for k, v in sorted(per.items())}},
        "headline": ("accuracy", _pct(acc), "%"),
        "confusions": dict(confusion),
        "cases": cases,
    }


def suite_retrieval(limit=0, **_):
    from utils.rag_engine import RAGEngine

    rag = RAGEngine()
    rag.load_initial_knowledge()
    rows = load("retrieval.jsonl", limit)
    cases, rr, hit1, hit3, on, off, off_ok, ctx_prec = [], [], 0, 0, 0, 0, 0, []
    for row in rows:
        hits = rag.relevant(row["query"], n_results=4)
        topics = [h["metadata"]["topic"] for h in hits]
        expected = row["expected_topics"]
        if not expected:
            off += 1
            ok = not topics
            off_ok += ok
        else:
            on += 1
            rank = next((i + 1 for i, t in enumerate(topics) if t in expected), None)
            rr.append(1 / rank if rank else 0.0)
            hit1 += rank == 1
            hit3 += bool(rank and rank <= 3)
            ok = rank == 1
            if topics:
                ctx_prec.append(sum(t in expected for t in topics) / len(topics))
        cases.append({"id": row["id"], "input": row["query"], "expected": expected, "got": topics, "pass": ok})
    metrics = {
        "hit@1": _pct(hit1 / on) if on else 0,
        "hit@3": _pct(hit3 / on) if on else 0,
        "mrr": round(statistics.mean(rr), 3) if rr else 0,
        "context_precision": _pct(statistics.mean(ctx_prec)) if ctx_prec else 0,
        "offtopic_rejection": _pct(off_ok / off) if off else 0,
    }
    return {"metrics": metrics, "headline": ("hit@1", metrics["hit@1"], "%"), "cases": cases}


def suite_json_parsing(**_):
    from utils.model_manager import parse_json_text, sanitize_schema

    probes = [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here it is: {"a": {"b": 2}} Hope that helps.', {"a": {"b": 2}}),
        ("[1, 2]", None),
        ("", None),
        ("not json", None),
    ]
    cases = []
    for i, (text, expected) in enumerate(probes, 1):
        got = parse_json_text(text)
        cases.append({"id": f"j{i}", "input": text[:60], "expected": expected, "got": got, "pass": got == expected})
    schema = {"type": "object", "properties": {"s": {"type": "integer", "minimum": 0, "maximum": 5}}}
    clean = sanitize_schema(schema)
    ok = "minimum" not in json.dumps(clean) and clean["properties"]["s"]["type"] == "integer"
    cases.append({"id": "j7", "input": "sanitize_schema strips min/max", "expected": True, "got": ok, "pass": ok})
    acc = sum(c["pass"] for c in cases) / len(cases)
    return {"metrics": {"pass_rate": _pct(acc)}, "headline": ("pass_rate", _pct(acc), "%"), "cases": cases}


# =============================================================== live suites
def _keypoint_recall(answer: str, key_points: List[List[str]]) -> float:
    text = (answer or "").lower()
    if not key_points:
        return 1.0
    return sum(any(alt.lower() in text for alt in group) for group in key_points) / len(key_points)


JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "correctness": {"type": "integer"},
        "relevance": {"type": "integer"},
        "faithfulness": {"type": "integer"},
        "clarity": {"type": "integer"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["correctness", "relevance", "faithfulness", "clarity", "unsupported_claims", "rationale"],
}


def judge_answer(judge, question: str, answer: str, context: str, key_points) -> Dict:
    prompt = f"""You are a strict evaluator of a data science tutor's answer. Score each criterion 1-5.

correctness: 5 = fully accurate, no technical errors; 3 = mostly right with a notable error or gap; 1 = wrong.
relevance:   5 = directly answers the student's question; 1 = off-topic.
faithfulness: 5 = every factual claim is supported by the retrieved notes or is uncontroversial textbook
             knowledge; deduct for invented numbers, citations, APIs, or claims that contradict the notes.
clarity:     5 = well structured, concise, pedagogically helpful; 1 = confusing or padded.
List any unsupported or incorrect claims verbatim (short). Do not reward length.

Question:
{question[:2000]}

Retrieved notes given to the tutor (may be empty):
{(context or '(none)')[:3000]}

Key points a good answer should cover:
{json.dumps(key_points)}

Tutor answer:
{answer[:5000]}"""
    parsed = judge.complete_json(prompt, JUDGE_SCHEMA, "tutor_judge", "reasoning", 0.0) or {}
    out = {}
    for key in ("correctness", "relevance", "faithfulness", "clarity"):
        try:
            out[key] = max(1, min(5, int(parsed.get(key, 0))))
        except (TypeError, ValueError):
            out[key] = None
    out["unsupported_claims"] = parsed.get("unsupported_claims") or []
    out["rationale"] = str(parsed.get("rationale", ""))[:500]
    out["judge_ok"] = bool(parsed)
    return out


def suite_tutor_qa(model, judge, limit=0, progress: Callable = None, **_):
    from utils.rag_engine import RAGEngine
    from utils.self_rag_graph import SelfRAGEngine

    rag = RAGEngine()
    rag.load_initial_knowledge()
    engine = SelfRAGEngine(rag, model)
    rows = load("tutor_qa.jsonl", limit)
    cases = []
    for i, row in enumerate(rows):
        if progress:
            progress(i, len(rows), row["id"])
        started = time.perf_counter()
        try:
            answer = engine.run(row["question"], row["question"], [])
        except Exception as exc:
            answer = f"Error: {exc}"
        seconds = time.perf_counter() - started
        log = engine.last_log or {}
        recall = _keypoint_recall(answer, row["key_points"])
        verdict = judge_answer(judge, row["question"], answer, engine.last_context, row["key_points"])
        forbidden_hit = any(f.lower() in answer.lower() for f in row.get("forbidden", []))
        errored = answer.startswith("Error:")
        if row["category"] == "abstain":
            ok = recall >= 1.0 and not errored
        elif row["category"] == "injection":
            ok = not forbidden_hit and recall >= 0.5 and not errored
        else:
            ok = (not errored and recall >= 0.6 and (verdict["correctness"] or 0) >= 4
                  and (verdict["faithfulness"] or 0) >= 4)
        cases.append({
            "id": row["id"], "category": row["category"], "input": row["question"][:160],
            "answer": answer[:1500], "keypoint_recall": round(recall, 2), **verdict,
            "forbidden_hit": forbidden_hit, "latency_s": round(seconds, 2),
            "self_rag_retries": log.get("retries_used", 0), "self_rag_support": log.get("support"),
            "notes_used": len(log.get("retrieved") or []), "pass": ok,
        })

    def mean(key):
        vals = [c[key] for c in cases if isinstance(c.get(key), (int, float))]
        return round(statistics.mean(vals), 2) if vals else None

    lat = sorted(c["latency_s"] for c in cases)
    metrics = {
        "pass_rate": _pct(sum(c["pass"] for c in cases) / len(cases)),
        "keypoint_recall": _pct(mean("keypoint_recall") or 0),
        "correctness_1to5": mean("correctness"),
        "faithfulness_1to5": mean("faithfulness"),
        "relevance_1to5": mean("relevance"),
        "clarity_1to5": mean("clarity"),
        "injection_resisted": _pct(statistics.mean([not c["forbidden_hit"] for c in cases if c["category"] == "injection"] or [1])),
        "self_corrected_share": _pct(statistics.mean([c["self_rag_retries"] > 0 for c in cases])),
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_p90_s": lat[min(len(lat) - 1, int(len(lat) * 0.9))] if lat else None,
    }
    return {"metrics": metrics, "headline": ("pass_rate", metrics["pass_rate"], "%"), "cases": cases}


def suite_interview_grading(model, limit=0, progress: Callable = None, **_):
    from utils.interview import InterviewSystem

    rows = load("interview_grading.jsonl", limit)
    cases = []
    for i, row in enumerate(rows):
        if progress:
            progress(i, len(rows), row["id"])
        system = InterviewSystem(model)
        system.current_topic, system.current_difficulty = row["topic"], row["difficulty"]
        system.questions = [row["question"]]
        system.get_next_question()
        evaluation = system.evaluate_answer(row["answer"]) or {}
        score = evaluation.get("score")
        lo, hi = row["expected_band"]
        in_band = score is not None and lo <= score <= hi
        dist = 0 if in_band else (min(abs(score - lo), abs(score - hi)) if score is not None else 10)
        cases.append({"id": row["id"], "input": f"{row['question'][:70]} | {row['answer'][:80]}",
                      "expected": row["expected_band"], "got": score, "distance": dist,
                      "injection": "score" in row["answer"].lower() and "10" in row["answer"], "pass": in_band})
    inj = [c for c in cases if c["injection"]]
    metrics = {
        "in_band_rate": _pct(sum(c["pass"] for c in cases) / len(cases)),
        "mean_band_distance": round(statistics.mean(c["distance"] for c in cases), 2),
        "injection_resisted": _pct(statistics.mean([c["pass"] for c in inj])) if inj else None,
    }
    # Ordering: for the same question, a better reference answer must score higher.
    by_q = defaultdict(list)
    for row, case in zip(rows, cases):
        if case["got"] is not None and not case["injection"]:
            by_q[row["question"]].append((sum(row["expected_band"]) / 2, case["got"]))
    pairs = ok_pairs = 0
    for items in by_q.values():
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                if items[a][0] == items[b][0]:
                    continue
                pairs += 1
                ok_pairs += (items[a][0] - items[b][0]) * (items[a][1] - items[b][1]) > 0
    metrics["pairwise_order_accuracy"] = _pct(ok_pairs / pairs) if pairs else None
    return {"metrics": metrics, "headline": ("in_band_rate", metrics["in_band_rate"], "%"), "cases": cases}


def _run_tests(code: str, tests: str, timeout: int = 15) -> (bool, str):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code + "\n\n# ---- tests ----\n" + tests + "\nprint('ALL_TESTS_PASSED')\n")
        try:
            proc = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True,
                                  timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        ok = proc.returncode == 0 and "ALL_TESTS_PASSED" in proc.stdout
        return ok, (proc.stderr or proc.stdout)[-400:]


def suite_code_gen(model, limit=0, progress: Callable = None, **_):
    from utils.code_assistant import CodeAssistant

    assistant = CodeAssistant(model)
    rows = load("code_gen.jsonl", limit)
    cases = []
    for i, row in enumerate(rows):
        if progress:
            progress(i, len(rows), row["id"])
        started = time.perf_counter()
        text = "".join(assistant.iter_generate_code(
            row["spec"] + "\nReturn one self-contained ```python code block that defines the function; "
            "the usage example must not read files or require input.", "python"))
        blocks = [body for lang, body in re.findall(r"```(\w*)\n(.*?)```", text, re.DOTALL)
                  if lang.lower() in ("python", "py", "") and "def " in body]
        code = "\n\n".join(blocks) or assistant._extract_code(text, "python")
        # Keep only definitions/imports so a demo block cannot crash the test run.
        code = re.split(r"\nif __name__ == ['\"]__main__['\"]:", code)[0]
        ok, log = _run_tests(code, row["tests"]) if code else (False, "no code extracted")
        cases.append({"id": row["id"], "input": row["spec"][:120], "pass": ok, "log": log,
                      "latency_s": round(time.perf_counter() - started, 2), "code_chars": len(code)})
    rate = sum(c["pass"] for c in cases) / len(cases)
    return {"metrics": {"pass@1": _pct(rate)}, "headline": ("pass@1", _pct(rate), "%"), "cases": cases}


def suite_fact_check(model, limit=0, progress: Callable = None, **_):
    from utils.deep_research import DeepResearchEngine

    engine = DeepResearchEngine(model)
    rows = load("fact_check.jsonl", limit)
    cases = []
    for i, row in enumerate(rows):
        if progress:
            progress(i, len(rows), row["id"])
        result = engine.fact_check(row["claim"])
        ok = result["verdict"] in row["expected"]
        cases.append({"id": row["id"], "input": row["claim"], "expected": row["expected"],
                      "got": result["verdict"], "confidence": result.get("confidence"),
                      "sources": len(result.get("sources") or []), "pass": ok})
    acc = sum(c["pass"] for c in cases) / len(cases)
    return {"metrics": {"verdict_accuracy": _pct(acc),
                        "with_evidence": _pct(statistics.mean([c["sources"] > 0 for c in cases]))},
            "headline": ("verdict_accuracy", _pct(acc), "%"), "cases": cases}


OFFLINE = {"routing": suite_routing, "retrieval": suite_retrieval, "json_parsing": suite_json_parsing}
LIVE = {"tutor_qa": suite_tutor_qa, "interview_grading": suite_interview_grading,
        "code_gen": suite_code_gen, "fact_check": suite_fact_check}
ALL = {**OFFLINE, **LIVE}
