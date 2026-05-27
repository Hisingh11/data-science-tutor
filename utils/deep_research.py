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
            results = [{
                "title": "Search Error",
                "body": f"Could not perform search: {str(e)}",
                "href": "",
                "source": "error"
            }]
        return results
    
    def fact_check(self, claim: str) -> Dict:
        search_results = self.search_web(claim, max_results=5)
        context = ""
        for result in search_results:
            context += f"Source: {result['title']}\nContent: {result['body']}\n\n"
        
        prompt = f"""Fact check this claim: {claim}

Evidence:
{context}

Provide response in JSON format:
{{
    "verdict": "true/false/partially_true/unverifiable",
    "confidence": 0-100,
    "evidence": "summary of evidence",
    "explanation": "detailed analysis"
}}"""
        
        response = self.model.generate(prompt, "reasoning", 0.2)
        
        try:
            import re
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "verdict": "unverifiable",
            "confidence": 50,
            "evidence": "Based on available search results",
            "explanation": response[:500],
            "sources": [r.get('href', '') for r in search_results if r.get('href')]
        }
    
    def deep_research(self, topic: str) -> Dict:
        search_queries = [
            f"{topic} overview",
            f"{topic} best practices",
            f"{topic} tutorial"
        ]
        
        all_results = {}
        for query in search_queries:
            all_results[query] = self.search_web(query, max_results=3)
        
        compilation = ""
        for query, results in all_results.items():
            compilation += f"\n\n=== {query} ===\n"
            for r in results:
                compilation += f"\n**{r['title']}**\n{r['body'][:500]}...\n"
        
        prompt = f"""Create a comprehensive research report on: {topic}

Research Data:
{compilation}

Provide:
1. Executive Summary
2. Key Concepts
3. Best Practices
4. Challenges and Solutions
5. Future Trends
6. Key Takeaways"""
        
        report = self.model.generate(prompt, "reasoning", 0.4)
        
        takeaways_prompt = f"List 5 key takeaways about {topic} for data scientists"
        takeaways = self.model.generate(takeaways_prompt, "fast", 0.3)
        
        return {
            "topic": topic,
            "report": report,
            "key_takeaways": takeaways,
            "sources": self._extract_sources(all_results),
            "searches_performed": list(all_results.keys())
        }
    
    def _extract_sources(self, all_results: Dict) -> List[Dict]:
        sources = []
        seen_urls = set()
        for results in all_results.values():
            for r in results:
                url = r.get('href', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        "title": r.get('title', ''),
                        "url": url,
                        "snippet": r.get('body', '')[:200]
                    })
        return sources[:10]