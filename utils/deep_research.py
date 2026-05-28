import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from duckduckgo_search import DDGS
from typing import List, Dict

class DeepResearchEngine:
    def __init__(self, model_manager):
        self.model = model_manager
        self.ddgs = DDGS()
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        results = []
        try:
            search_results = list(self.ddgs.text(query, max_results=max_results))
            for result in search_results:
                results.append({
                    "title": result.get('title', ''),
                    "body": result.get('body', ''),
                    "href": result.get('href', ''),
                    "source": "duckduckgo"
                })
        except Exception as e:
            results = [{"title": "Search Error", "body": str(e), "href": "", "source": "error"}]
        return results
    
    def fact_check(self, claim: str) -> Dict:
        search_results = self.search_web(claim, max_results=5)
        context = ""
        for result in search_results:
            context += f"Source: {result['title']}\nContent: {result['body']}\n\n"
        prompt = f"Fact check this claim: {claim}\n\nEvidence: {context}\n\nReturn JSON with verdict, confidence, explanation"
        response = self.model.generate(prompt, "reasoning", 0.2)
        try:
            import re, json
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return {"verdict": "unverifiable", "confidence": 50, "evidence": "Based on search", "explanation": response[:500]}
    
    def deep_research(self, topic: str) -> Dict:
        queries = [f"{topic} overview", f"{topic} best practices", f"{topic} tutorial"]
        all_results = {}
        for q in queries:
            all_results[q] = self.search_web(q, max_results=3)
        compilation = ""
        for q, results in all_results.items():
            compilation += f"\n\n=== {q} ===\n"
            for r in results:
                compilation += f"\n**{r['title']}**\n{r['body'][:300]}...\n"
        prompt = f"Research report on {topic}:\n{compilation}\n\nInclude summary, key concepts, best practices, challenges, takeaways"
        report = self.model.generate(prompt, "reasoning", 0.4)
        takeaways = self.model.generate(f"5 key takeaways about {topic}", "fast", 0.3)
        return {"topic": topic, "report": report, "key_takeaways": takeaways, "sources": [], "searches_performed": list(all_results.keys())}
    
    def _extract_sources(self, all_results: Dict) -> List[Dict]:
        sources = []
        seen = set()
        for results in all_results.values():
            for r in results:
                url = r.get('href', '')
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"title": r.get('title', ''), "url": url})
        return sources[:10]