import json
import re
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore", category=RuntimeWarning)


class DeepResearchEngine:
    def __init__(self, model_manager):
        self.model = model_manager

    def search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        try:
            from ddgs import DDGS
        except ImportError:
            return [{
                "title": "Search unavailable",
                "body": "Install the ddgs package to enable web research.",
                "href": "",
                "source": "error",
            }]
        try:
            search_results = DDGS().text(query, max_results=max_results) or []
            results = []
            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "href": result.get("href", ""),
                    "source": "web",
                })
            if not results:
                return [{
                    "title": "No results",
                    "body": "The search returned nothing for this query.",
                    "href": "",
                    "source": "empty",
                }]
            return results
        except Exception as exc:
            return [{
                "title": "Search error",
                "body": str(exc),
                "href": "",
                "source": "error",
            }]

    def fact_check(self, claim: str) -> Dict:
        search_results = self.search_web(claim[:300], max_results=5)
        context = "\n\n".join(
            f"Source: {result['title']}\nContent: {result['body']}"
            for result in search_results
        )
        prompt = (
            f"Fact-check this claim:\n{claim}\n\nEvidence:\n{context}\n\n"
            "Return JSON with keys verdict, confidence, explanation. "
            "verdict must be one of supported, contradicted, mixed, unverifiable."
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "verdict": {"type": "string"},
                "confidence": {"type": "integer"},
                "explanation": {"type": "string"},
            },
            "required": ["verdict", "confidence", "explanation"],
        }
        parsed = None
        if hasattr(self.model, "complete_json"):
            parsed = self.model.complete_json(prompt, schema, "fact_check", "reasoning", 0.2)
        if not isinstance(parsed, dict):
            response = self.model.generate(prompt, "reasoning", 0.2)
            try:
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    parsed.setdefault("explanation", response[:800])
            except Exception:
                parsed = {
                    "verdict": "unverifiable",
                    "confidence": 50,
                    "explanation": response[:800],
                }
        if not isinstance(parsed, dict):
            parsed = {"verdict": "unverifiable", "confidence": 50, "explanation": ""}
        verdict = str(parsed.get("verdict", "unverifiable")).strip().lower()
        if verdict not in {"supported", "contradicted", "mixed", "unverifiable"}:
            verdict = "unverifiable"
        parsed["verdict"] = verdict
        return parsed

    def deep_research(self, topic: str, context: str = "") -> Dict:
        queries = [
            f"{topic} overview",
            f"{topic} how it works",
            f"{topic} compared with alternatives",
            f"{topic} limitations and failure cases",
            f"{topic} practical workflow",
        ]
        all_results = {query: self.search_web(query, max_results=4) for query in queries}
        compilation = ""
        for query, results in all_results.items():
            compilation += f"\n\n=== {query} ===\n"
            for result in results:
                compilation += f"\n**{result['title']}**\n{result['body'][:400]}\n"
        prompt = (
            f"Write an advanced research briefing on {topic} for a data science student.\n"
            f"Source notes:\n{compilation[:12000]}\n\n"
            f"Extra context from the user:\n{(context or '')[:4000]}\n\n"
            "Use these sections: what it is, how it works, where it beats the alternatives, "
            "a concrete worked example, failure cases, and what to study next. "
            "Stay tied to the notes. Do not invent citations or numbers."
        )
        report = self.model.generate(prompt, "reasoning", 0.4)
        takeaways = self.model.generate(
            f"Give 5 short study takeaways about {topic}. Use a numbered list.",
            "fast",
            0.3,
        )
        return {
            "topic": topic,
            "report": report,
            "key_takeaways": takeaways,
            "sources": self._extract_sources(all_results),
            "searches_performed": list(all_results.keys()),
        }

    def _extract_sources(self, all_results: Dict) -> List[Dict]:
        sources = []
        seen = set()
        for results in all_results.values():
            for result in results:
                url = result.get("href", "")
                if url and url not in seen and result.get("source") not in ("error", "empty"):
                    seen.add(url)
                    sources.append({"title": result.get("title", ""), "url": url})
        return sources[:10]
