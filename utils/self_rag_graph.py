"""LangGraph Self-RAG workflow for tutor chat."""

from threading import Event
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


MAX_RETRIES = 1
ANSWER_TEMPERATURE = 0.85
RETRIEVAL_TEMPERATURE = 0.35
EVALUATION_TEMPERATURE = 0.35

# Enable/disable web search to reduce latency
ENABLE_WEB_SEARCH = False


class SelfRAGState(TypedDict, total=False):
    query: str
    user_content: str
    history: Optional[List[Dict]]
    cancel: Optional[Event]
    candidates: List[Dict]
    local_notes: List[Dict]
    web_candidates: List[Dict]
    web_notes: List[Dict]
    context: str
    web: int
    retrieved: List[Dict]
    answer: str
    feedback: str
    retry_count: int
    calls: int
    needs_revision: bool
    support: str
    usefulness: str
    revised: bool


class SelfRAGEngine:
    """Retrieve, answer, and self-correct within the configured retry limit."""

    def __init__(self, rag, model, max_retries: int = MAX_RETRIES):
        self.rag = rag
        self.model = model
        self.max_retries = min(MAX_RETRIES, max(0, int(max_retries)))
        self.last_log: Optional[Dict] = None
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(SelfRAGState)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("combine_context", self._combine_context)
        workflow.add_node("generate", self._generate)
        workflow.add_node("reflect", self._reflect)
        workflow.add_node("prepare_retry", self._prepare_retry)

        if ENABLE_WEB_SEARCH:
            workflow.add_node("web_search", self._web_search)
            workflow.add_node("grade_web", self._grade_web)
            workflow.add_edge(START, "retrieve")
            workflow.add_edge("retrieve", "web_search")
            workflow.add_edge("web_search", "grade_documents")
            workflow.add_edge("grade_documents", "grade_web")
            workflow.add_edge("grade_web", "combine_context")
        else:
            workflow.add_edge(START, "retrieve")
            workflow.add_edge("retrieve", "grade_documents")
            workflow.add_edge("grade_documents", "combine_context")

        workflow.add_edge("combine_context", "generate")
        workflow.add_edge("generate", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self._after_reflection,
            {"retry": "prepare_retry", "end": END},
        )
        workflow.add_edge("prepare_retry", "generate")
        return workflow.compile()

    def iter_answer(self, query: str, user_content: str, history=None, cancel=None):
        """Run the graph and yield the completed answer in display-sized chunks."""
        self.last_log = None
        state: SelfRAGState = {
            "query": query,
            "user_content": user_content,
            "history": history,
            "cancel": cancel,
            "candidates": [],
            "local_notes": [],
            "web_candidates": [],
            "web_notes": [],
            "context": "",
            "web": 0,
            "retrieved": [],
            "answer": "",
            "feedback": "",
            "retry_count": 0,
            "calls": 0,
            "needs_revision": False,
            "support": "not checked",
            "usefulness": "not checked",
            "revised": False,
        }
        try:
            result = self.graph.invoke(state, config={"recursion_limit": 32})
            answer = result.get("answer", "")
            self.last_log = {
                "llm_checks": result.get("calls", 0),
                "retries_used": result.get("retry_count", 0),
                "max_retries": self.max_retries,
                "retrieved": result.get("retrieved", []),
                "web": result.get("web", 0),
                "support": result.get("support", "not checked"),
                "usefulness": result.get("usefulness", "not checked"),
                "revised": result.get("revised", False),
            }
        except Exception:
            self.last_log = {
                "llm_checks": state.get("calls", 0),
                "retries_used": state.get("retry_count", 0),
                "max_retries": self.max_retries,
                "retrieved": state.get("retrieved", []),
                "web": state.get("web", 0),
                "support": "not checked",
                "usefulness": "not checked",
                "revised": state.get("revised", False),
            }
            raise
        if answer:
            yield from self._chunks(answer)

    def _retrieve(self, state: SelfRAGState) -> SelfRAGState:
        if self._cancelled(state):
            return state
        query = state.get("query", "")
        state["candidates"] = self.rag.search(query, n_results=5) if query.strip() else []
        return state

    def _grade_documents(self, state: SelfRAGState) -> SelfRAGState:
        candidates = state.get("candidates", [])
        if self._cancelled(state) or not candidates:
            return state

        selected = [
            {
                "topic": candidate["metadata"].get("topic", "note"),
                "text": candidate["text"][:400],
            }
            for candidate in candidates
            if candidate.get("relevance", 0) >= 0.3
        ]

        state["local_notes"] = selected
        state["retrieved"] = [
            {"topic": item["topic"], "source": "knowledge"} for item in selected
        ]
        state["context"] = "\n\n".join(
            f"[{item['topic']}]\n{item['text']}" for item in selected
        )
        return state

    def _web_search(self, state: SelfRAGState) -> SelfRAGState:
        if self._cancelled(state):
            return state
        state["web_candidates"] = self.rag.web_notes(
            state.get("query", ""), max_results=5
        ) or []
        return state

    def _grade_web(self, state: SelfRAGState) -> SelfRAGState:
        candidates = state.get("web_candidates", [])
        if self._cancelled(state) or not candidates:
            return state

        numbered = "\n\n".join(
            f"{index}. {item.get('title', 'Web result')}\n"
            f"URL: {item.get('href', '')}\n{item.get('body', '')[:500]}"
            for index, item in enumerate(candidates, start=1)
        )
        prompt = (
            "You are a web retrieval critic for a data science tutor. Judge whether each "
            "search result directly helps answer the question. Prefer primary or reliable "
            "sources and reject vague, duplicate, promotional, or off-topic snippets.\n\n"
            f"Question: {state['query']}\n\nSearch results:\n{numbered}\n\n"
            "For each result, set score from 0 (irrelevant) to 5 (directly answers the question "
            "and comes from a reliable source). Mark relevant yes only when score is at least 3. "
            "Return only the useful evidence in strip; never add facts that are not in the snippet."
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "grades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "integer"},
                            "relevant": {"type": "string", "enum": ["yes", "no"]},
                            "score": {"type": "integer", "minimum": 0, "maximum": 5},
                            "strip": {"type": "string"},
                        },
                        "required": ["id", "relevant", "score", "strip"],
                    },
                }
            },
            "required": ["grades"],
        }
        state["calls"] = state.get("calls", 0) + 1
        parsed = self.model.complete_json(
            prompt, schema, "selfrag_web_relevance", "fast", RETRIEVAL_TEMPERATURE
        )
        grades = parsed.get("grades") if isinstance(parsed, dict) else None
        selected = []
        if isinstance(grades, list):
            for item in grades:
                if not isinstance(item, dict) or str(item.get("relevant", "")).lower() != "yes":
                    continue
                try:
                    index = int(item.get("id", 0)) - 1
                except (TypeError, ValueError):
                    continue
                if not 0 <= index < len(candidates):
                    continue
                try:
                    score = int(item.get("score", 0))
                except (TypeError, ValueError):
                    continue
                if score < 3:
                    continue
                candidate = candidates[index]
                selected.append({
                    "title": candidate.get("title", "Web result"),
                    "href": candidate.get("href", ""),
                    "text": str(item.get("strip") or candidate.get("body", ""))[:500],
                    "score": score,
                })
        elif not isinstance(grades, list):
            for candidate in candidates:
                body = candidate.get("body", "")
                if self.rag._anchored(state["query"], body):
                    selected.append({
                        "title": candidate.get("title", "Web result"),
                        "href": candidate.get("href", ""),
                        "text": body[:500],
                        "score": 3,
                    })

        selected.sort(key=lambda item: item["score"], reverse=True)
        state["web_notes"] = selected[:3]
        state["web"] = len(state["web_notes"])
        state["retrieved"] = state.get("retrieved", []) + [
            {"topic": item["title"], "source": "web", "href": item["href"]}
            for item in state["web_notes"]
        ]
        return state

    def _combine_context(self, state: SelfRAGState) -> SelfRAGState:
        parts = [
            f"[Knowledge base: {item['topic']}]\n{item['text']}"
            for item in state.get("local_notes", [])
        ]
        for item in state.get("web_notes", []):
            href = f"\nURL: {item['href']}" if item.get("href") else ""
            parts.append(f"[Web source: {item['title']}]{href}\n{item['text']}")
        if state.get("web_notes"):
            parts.insert(
                0,
                "Web results are search snippets, not independently verified pages. "
                "Use only evidence that directly supports the answer and cite provided URLs.",
            )
        state["context"] = "\n\n".join(parts)
        return state

    def _generate(self, state: SelfRAGState) -> SelfRAGState:
        if self._cancelled(state):
            return state
        prompt = (
            f"{state['user_content']}\n\n"
            "Keep the response focused and proportional to the question. "
            "Avoid repeating explanations."
        )
        if state.get("retry_count", 0):
            prompt = (
                f"Student request:\n{prompt[:1500]}\n\n"
                f"Previous draft:\n{state.get('answer', '')[:2200]}\n\n"
                f"Self-check feedback:\n{state.get('feedback', '')[:1200]}\n\n"
                "Write a corrected, complete answer. Fix unsupported claims and address "
                "the listed gaps. Return only the answer."
            )
        state["calls"] = state.get("calls", 0) + 1
        state["answer"] = self.model.answer(
            prompt,
            history=state.get("history"),
            context=state.get("context", ""),
            model_type="reasoning",
            temperature=ANSWER_TEMPERATURE,
            max_tokens=2048,
        )
        state["revised"] = state.get("retry_count", 0) > 0
        state["needs_revision"] = False
        return state

    def _reflect(self, state: SelfRAGState) -> SelfRAGState:
        answer = state.get("answer", "")
        if self._cancelled(state) or self._model_error(answer):
            return state
        prompt = (
            "Evaluate this tutor answer. Judge factual claims against all supplied retrieved "
            "evidence, including local knowledge and web sources, and judge whether it answers "
            "the student's request.\n\n"
            f"Question: {state['query']}\n\n"
            f"Knowledge notes:\n{state.get('context', '')[:2000]}\n\n"
            f"Answer:\n{answer[:2200]}\n\n"
            "List up to 8 factual claims, marking each fully, partially, unsupported, "
            "or not_a_claim. Set useful to high, medium, or low. List important missing "
            "points; use an empty list when complete."
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "claim": {"type": "string"},
                            "support": {
                                "type": "string",
                                "enum": ["fully", "partially", "unsupported", "not_a_claim"],
                            },
                        },
                        "required": ["claim", "support"],
                    },
                },
                "useful": {"type": "string", "enum": ["high", "medium", "low"]},
                "missing": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["claims", "useful", "missing"],
        }
        state["calls"] = state.get("calls", 0) + 1
        parsed = self.model.complete_json(
            prompt, schema, "selfrag_reflection", "fast", EVALUATION_TEMPERATURE
        )
        if not isinstance(parsed, dict):
            return state

        claims = parsed.get("claims")
        if not isinstance(claims, list):
            claims = []
        unsupported = [
            str(item.get("claim", ""))[:200]
            for item in claims
            if isinstance(item, dict) and item.get("support") == "unsupported"
        ]
        useful = str(parsed.get("useful", "medium")).lower()
        if useful not in {"high", "medium", "low"}:
            useful = "medium"
        missing_items = parsed.get("missing")
        if not isinstance(missing_items, list):
            missing_items = []
        missing = [str(item)[:200] for item in missing_items[:5] if str(item).strip()]
        has_sources = bool(state.get("local_notes") or state.get("web_notes"))
        state["support"] = (
            "no relevant sources" if not has_sources
            else f"{len(unsupported)} unsupported claim(s)" if unsupported
            else "claims grounded"
        )
        state["usefulness"] = f"{useful} ({len(missing)} gap(s))" if missing else useful
        state["needs_revision"] = (
            (has_sources and bool(unsupported))
            or useful == "low"
            or (useful == "medium" and bool(missing))
        )
        state["feedback"] = (
            f"Unsupported claims: {'; '.join(unsupported) or 'none'}\n"
            f"Missing points: {'; '.join(missing) or 'none'}"
        )
        return state

    def _prepare_retry(self, state: SelfRAGState) -> SelfRAGState:
        state["retry_count"] = state.get("retry_count", 0) + 1
        return state

    def _after_reflection(self, state: SelfRAGState) -> str:
        if self._cancelled(state):
            return "end"
        if state.get("needs_revision") and state.get("retry_count", 0) < self.max_retries:
            return "retry"
        return "end"

    @staticmethod
    def _cancelled(state: SelfRAGState) -> bool:
        cancel = state.get("cancel")
        return bool(cancel is not None and cancel.is_set())

    @staticmethod
    def _model_error(answer: str) -> bool:
        return answer.startswith(("Error:", "API key not configured", "GROQ_API_KEY"))

    @staticmethod
    def _chunks(text: str, step: int = 48):
        for start in range(0, len(text), step):
            yield text[start:start + step]