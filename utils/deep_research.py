import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List


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
        search_results = [r for r in self.search_web(claim[:300], max_results=5) if r.get("source") == "web"]
        context = "\n\n".join(
            f"Source: {result['title']} ({result.get('href', '')})\nContent: {result['body']}"
            for result in search_results
        ) or "(No web evidence was found.)"
        prompt = (
            f"Fact-check this claim:\n{claim}\n\nEvidence:\n{context}\n\n"
            "Return JSON with keys verdict, confidence, explanation. "
            "verdict must be one of supported, contradicted, mixed, unverifiable. "
            "confidence is an integer from 0 to 100. Use unverifiable when the evidence "
            "does not address the claim."
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
            response = self.model.generate(prompt, "reasoning", 0.2) or ""
            try:
                match = re.search(r"\{.*\}", response, re.DOTALL)
                parsed = json.loads(match.group()) if match else None
                if isinstance(parsed, dict):
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
        try:
            parsed["confidence"] = max(0, min(100, int(parsed.get("confidence", 50))))
        except (TypeError, ValueError):
            parsed["confidence"] = 50
        parsed["sources"] = [
            {"title": r.get("title", ""), "url": r.get("href", "")} for r in search_results if r.get("href")
        ][:5]
        return parsed

    def _prepare_research(self, topic: str, context: str = "", cancel=None) -> Dict:
        queries = [
            f"{topic} overview",
            f"{topic} how it works",
            f"{topic} compared with alternatives",
            f"{topic} limitations and failure cases",
            f"{topic} practical workflow",
        ]
        all_results = {}
        if not (cancel is not None and cancel.is_set()):
            # The five searches are independent, so run them together (was sequential).
            with ThreadPoolExecutor(max_workers=5) as pool:
                found = list(pool.map(lambda q: self.search_web(q, max_results=4), queries))
            all_results = dict(zip(queries, found))
        compilation = ""
        for query, results in all_results.items():
            usable = [r for r in results if r.get("source") == "web"]
            if not usable:
                continue
            compilation += f"\n\n=== {query} ===\n"
            for result in usable:
                compilation += f"\n**{result['title']}** ({result.get('href', '')})\n{result['body'][:400]}\n"
        if not compilation.strip():
            compilation = "(Web search returned no usable results. Say so, then answer from general knowledge and flag it as unverified.)"
        prompt = (
            f"Write an advanced research briefing on {topic} for a data science student.\n"
            f"Source notes:\n{compilation[:8000]}\n\n"
            f"Extra context from the user:\n{(context or '')[:2000]}\n\n"
            "Use these sections: what it is, how it works, where it beats the alternatives, "
            "a concrete worked example, failure cases, and what to study next. "
            "Stay tied to the notes. Do not invent citations or numbers."
        )
        return {
            "topic": topic,
            "prompt": prompt,
            "takeaways_prompt": f"Give 5 short study takeaways about {topic}. Use a numbered list.",
            "sources": self._extract_sources(all_results),
        }

    def iter_prepared(self, prepared: Dict):
        """Yield a briefing that was already searched. Tokens arrive as the model writes them."""
        yield f"**{prepared['topic']}**\n\n"
        yield from self.model.stream_answer(
            prepared["takeaways_prompt"], model_type="fast", temperature=0.3
        )
        yield "\n\n"
        yield from self.model.stream_answer(prepared["prompt"], temperature=0.4)
        sources = prepared["sources"]
        if sources:
            lines = "\n".join(f"- [{item['title']}]({item['url']})" for item in sources)
            yield "\n\n**Sources**\n" + lines

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
