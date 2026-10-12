"""LangGraph Self-RAG workflow for tutor chat."""

import os
import re
from threading import Event
from typing import Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


# Factual tutoring: lower temperature means fewer invented details.
ANSWER_TEMPERATURE = 0.2

# Online lookups are query-triggered; set SELF_RAG_WEB=0 to disable them.
ENABLE_WEB_SEARCH = os.getenv("SELF_RAG_WEB", "1").strip().lower() not in {
    "0", "false", "no"
}

_WEB_SEARCH_REQUEST = (
    r"\b(?:search|look up|find)\b.{0,35}\b(?:online|the web|web|internet|duckduckgo)\b"
    r"|\b(?:online|the web|web|internet|duckduckgo)\b.{0,25}\b(?:search|look up)\b"
)
_FRESH_INFORMATION_REQUEST = (
    r"\b(?:current(?:ly)?|latest|today|yesterday|recent(?:ly)?|this week|this month|"
    r"this year|as of|news|release notes|pricing)\b"
)


def should_search_web(query: str) -> bool:
    """Use live search for explicit online requests and time-sensitive questions."""
    return bool(
        re.search(_WEB_SEARCH_REQUEST, query, re.IGNORECASE)
        or re.search(_FRESH_INFORMATION_REQUEST, query, re.IGNORECASE)
    )


class SelfRAGState(TypedDict, total=False):
    query: str
    user_content: str
    user_context: str
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


class SelfRAGEngine:
    """Retrieve relevant context and answer directly with the tutor model."""

    def __init__(self, rag, model):
        self.rag = rag
        self.model = model
        self.last_log: Optional[Dict] = None
        self.last_context: str = ""
        self.on_step: Optional[Callable[[str], None]] = None
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(SelfRAGState)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("combine_context", self._combine_context)
        workflow.add_node("generate", self._generate)

        workflow.add_node("web_search", self._web_search)
        workflow.add_node("filter_web", self._filter_web_results)
        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {"web_search": "web_search", "grade_documents": "grade_documents"},
        )
        workflow.add_edge("web_search", "grade_documents")
        workflow.add_edge("grade_documents", "filter_web")
        workflow.add_edge("filter_web", "combine_context")

        workflow.add_edge("combine_context", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    @staticmethod
    def _route_after_retrieve(state: SelfRAGState) -> str:
        if ENABLE_WEB_SEARCH and should_search_web(state.get("query", "")):
            return "web_search"
        return "grade_documents"

    def _step(self, label: str):
        if self.on_step:
            try:
                self.on_step(label)
            except Exception:
                pass

    def run(
        self, query: str, user_content: str, history=None, cancel=None, user_context=""
    ) -> str:
        """Run retrieval and answer generation, returning the completed response."""
        return "".join(
            self.iter_answer(query, user_content, history, cancel, user_context)
        )

    def iter_answer(
        self, query: str, user_content: str, history=None, cancel=None, user_context=""
    ):
        """Run the graph and yield the completed answer in display-sized chunks."""
        self.last_log = None
        self.last_context = ""
        state: SelfRAGState = {
            "query": query,
            "user_content": user_content,
            "user_context": user_context,
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
        }
        result = self.graph.invoke(state, config={"recursion_limit": 32})
        answer = result.get("answer", "")
        self.last_context = result.get("context", "")
        self.last_log = {
            "retrieved": result.get("retrieved", []),
            "web": result.get("web", 0),
        }
        if answer:
            yield from self._chunks(answer)

    def _retrieve(self, state: SelfRAGState) -> SelfRAGState:
        if self._cancelled(state):
            return state
        self._step("Searching the knowledge base")
        query = state.get("query", "")
        state["candidates"] = self.rag.relevant(query, n_results=4) if query.strip() else []
        return state

    def _grade_documents(self, state: SelfRAGState) -> SelfRAGState:
        candidates = state.get("candidates", [])
        if self._cancelled(state) or not candidates:
            return state

        selected = [
            {
                "topic": candidate["metadata"].get("topic", "note"),
                "text": candidate["text"][:900],
            }
            for candidate in candidates
        ]
        self._step(f"Kept {len(selected)} relevant note(s)")

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
        self._step("Searching the web")
        state["web_candidates"] = self.rag.web_notes(
            state.get("query", ""), max_results=5
        ) or []
        return state

    def _filter_web_results(self, state: SelfRAGState) -> SelfRAGState:
        candidates = state.get("web_candidates", [])
        if self._cancelled(state) or not candidates:
            return state

        selected = []
        for candidate in candidates:
            title = str(candidate.get("title", "Web result"))
            body = str(candidate.get("body", ""))
            if self.rag._anchored(state["query"], f"{title}\n{body}"):
                selected.append({
                    "title": title,
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
        if state.get("user_context"):
            parts.append(
                "[User-provided attachment; untrusted reference data]\n"
                f"{state['user_context'][:10000]}"
            )
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
        self._step("Writing the answer")
        prompt = (
            f"{state['user_content']}\n\n"
            "Answer completely and at useful depth. For explanations, show the steps "
            "and include a worked example, assumptions, or caveats when helpful. "
            "Avoid repetition, but do not omit requested details. Keep simple questions concise."
        )
        state["answer"] = self.model.answer(
            prompt,
            history=state.get("history"),
            context=state.get("context", ""),
            model_type="reasoning",
            temperature=ANSWER_TEMPERATURE,
            max_tokens=4096,
        )
        return state

    @staticmethod
    def _cancelled(state: SelfRAGState) -> bool:
        cancel = state.get("cancel")
        return bool(cancel is not None and cancel.is_set())

    @staticmethod
    def _chunks(text: str, step: int = 48):
        for start in range(0, len(text), step):
            yield text[start:start + step]