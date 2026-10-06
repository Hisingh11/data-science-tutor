"""Run the eval suites and write JSON + Markdown reports.

Examples (from the project root):
  python -m evals.run_evals --offline                 # no API key needed
  python -m evals.run_evals                           # everything (uses Groq)
  python -m evals.run_evals --suites tutor_qa --limit 5
  python -m evals.run_evals --judge-model openai/gpt-oss-20b
  python -m evals.run_evals --mock                    # harness smoke test, fake model

Results land in evals/results/latest.json and evals/results/latest_report.md,
plus a timestamped copy. The LLM Evals page in the app reads latest.json.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evals.suites import ALL, LIVE, OFFLINE  # noqa: E402

RESULTS = os.path.join(ROOT, "evals", "results")

THRESHOLDS = {  # headline metric gates used for PASS/FAIL in the report
    "routing": 90, "retrieval": 85, "json_parsing": 100,
    "tutor_qa": 75, "interview_grading": 70, "code_gen": 66, "fact_check": 75,
}


def build_models(mock: bool, judge_model: str = ""):
    if mock:
        from evals.mock_model import MockModel

        model = MockModel()
        return model, model
    from utils.model_manager import ModelManager

    model = ModelManager()
    if not model.ready:
        raise SystemExit(f"Live evals need Groq: {model.init_error}")
    judge = ModelManager()
    judge_model = judge_model or os.getenv("EVAL_JUDGE_MODEL", "")
    if judge_model:
        judge.models = {k: judge_model for k in judge.models}
    return model, judge


def run(suites, limit=0, mock=False, judge_model="", progress=None, log=print):
    started = time.time()
    model = judge = None
    if any(name in LIVE for name in suites):
        model, judge = build_models(mock, judge_model)
    report = {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "mock" if mock else "live" if model else "offline",
        "models": getattr(model, "models", None),
        "judge_model": (getattr(judge, "models", {}) or {}).get("reasoning") if judge else None,
        "limit": limit,
        "suites": {},
    }
    for name in suites:
        log(f"▶ {name}")
        t0 = time.time()

        def tick(i, n, case_id, _name=name):
            if progress:
                progress(_name, i, n, case_id)
            log(f"   {_name} {i + 1}/{n} {case_id}")

        try:
            result = ALL[name](model=model, judge=judge, limit=limit, progress=tick)
            result["seconds"] = round(time.time() - t0, 1)
            metric, value, unit = result["headline"]
            result["threshold"] = THRESHOLDS.get(name)
            result["status"] = "pass" if value is not None and value >= THRESHOLDS.get(name, 0) else "fail"
            log(f"   {metric} = {value}{unit}  [{result['status'].upper()}]")
        except SystemExit:
            raise
        except Exception as exc:
            result = {"status": "error", "error": repr(exc), "metrics": {}, "cases": []}
            log(f"   ERROR {exc!r}")
        result["run_at"] = report["run_at"]
        result["mode"] = report["mode"]
        report["suites"][name] = result
    if model is not None:
        report["usage"] = model.usage_summary()
        if judge is not model:
            report["judge_usage"] = judge.usage_summary()
    report["seconds"] = round(time.time() - started, 1)
    return report


def to_markdown(report) -> str:
    lines = [f"# Eval report — {report['run_at']}", "",
             f"Mode: **{report['mode'].upper()}**" + (f" · models: `{report['models']}`" if report.get("models") else "")
             + (f" · judge: `{report['judge_model']}`" if report.get("judge_model") else ""), ""]
    if report["mode"] == "mock":
        lines += ["> MOCK run: a fake model was used to test the harness. Scores say nothing about real quality.", ""]
    lines += ["| Suite | Headline | Threshold | Status | Time / run |", "|---|---|---|---|---|"]
    for name, res in report["suites"].items():
        head = res.get("headline")
        value = f"{head[0]} = {head[1]}{head[2]}" if head else res.get("error", "")
        lines.append(f"| {name} | {value} | {res.get('threshold', '')} | {res['status'].upper()} | {res.get('seconds', '')}s · {res.get('mode', '')} {res.get('run_at', '')} |")
    for name, res in report["suites"].items():
        lines += ["", f"## {name}", ""]
        for key, value in (res.get("metrics") or {}).items():
            lines.append(f"- **{key}**: {value}")
        failed = [c for c in res.get("cases", []) if not c.get("pass")]
        if failed:
            lines += ["", f"Failures ({len(failed)}):", ""]
            for case in failed[:15]:
                detail = {k: v for k, v in case.items() if k in ("expected", "got", "keypoint_recall", "correctness",
                                                                  "faithfulness", "log", "rationale")}
                lines.append(f"- `{case['id']}` {str(case.get('input', ''))[:90]!r} → {json.dumps(detail, ensure_ascii=False)[:300]}")
    return "\n".join(lines) + "\n"


def merge_with_latest(report):
    """Keep earlier suite results in latest.json when only some suites were re-run."""
    if report["mode"] == "mock":
        return report
    try:
        with open(os.path.join(RESULTS, "latest.json"), encoding="utf-8") as handle:
            previous = json.load(handle)
    except (OSError, ValueError):
        return report
    if previous.get("mode") == "mock":
        return report
    merged = {**previous.get("suites", {}), **report["suites"]}
    modes = {res.get("mode", previous.get("mode")) for res in merged.values()}
    return {**report, "suites": {name: merged[name] for name in ALL if name in merged},
            "mode": "live" if "live" in modes else report["mode"],
            "models": report.get("models") or previous.get("models"),
            "judge_model": report.get("judge_model") or previous.get("judge_model")}


def save(report):
    os.makedirs(RESULTS, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(RESULTS, f"{stamp}_{report['mode']}.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
    report = merge_with_latest(report)
    with open(os.path.join(RESULTS, "latest.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(RESULTS, "latest_report.md"), "w", encoding="utf-8") as handle:
        handle.write(to_markdown(report))
    return os.path.join(RESULTS, "latest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites", nargs="*", default=None, help=f"subset of {list(ALL)}")
    parser.add_argument("--offline", action="store_true", help="only the deterministic suites")
    parser.add_argument("--mock", action="store_true", help="use a fake model (harness smoke test)")
    parser.add_argument("--limit", type=int, default=0, help="max cases per suite (0 = all)")
    parser.add_argument("--judge-model", default="", help="Groq model id for LLM-as-judge")
    args = parser.parse_args()
    suites = args.suites or (list(OFFLINE) if args.offline else list(ALL))
    unknown = [s for s in suites if s not in ALL]
    if unknown:
        parser.error(f"unknown suites {unknown}; choose from {list(ALL)}")
    report = run(suites, args.limit, args.mock, args.judge_model)
    path = save(report)
    print(f"\nSaved {path}")
    print(to_markdown(report).split("\n## ")[0])
    sys.exit(0 if all(r["status"] == "pass" for r in report["suites"].values()) else 1)


if __name__ == "__main__":
    main()
