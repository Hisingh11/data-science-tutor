"""Shared objects and session helpers for every page."""

import json
import os
import time
import uuid
from types import SimpleNamespace

import streamlit as st

from utils.assignment import AssignmentManager
from utils.code_assistant import CodeAssistant
from utils.deep_research import DeepResearchEngine
from utils.interview import InterviewSystem
from utils.model_manager import ModelManager
from utils.rag_engine import RAGEngine
from utils.self_rag_graph import SelfRAGEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_PATH = os.path.join(ROOT, "data", "feedback.jsonl")
UPLOAD_DIR = os.path.join(ROOT, "data", "uploads")
ASSIGNMENTS_DIR = os.path.join(ROOT, "assignments")


@st.cache_resource(show_spinner=False)
def shared_model() -> ModelManager:
    """One Groq client per server process (cleared by 'Reload API key').

    DSBOT_MOCK=1 swaps in the offline fake model, for UI work without a key.
    """
    if os.getenv("DSBOT_MOCK") == "1":
        from evals.mock_model import MockModel

        return MockModel()
    return ModelManager()


@st.cache_resource(show_spinner=False)
def shared_rag() -> RAGEngine:
    rag = RAGEngine()
    rag.load_initial_knowledge()
    return rag


def core() -> SimpleNamespace:
    """Per-session tools. Interview and assignment state must not leak between users."""
    if "core" not in st.session_state:
        model = shared_model()
        st.session_state.core = SimpleNamespace(
            model=model,
            rag=shared_rag(),
            self_rag=SelfRAGEngine(shared_rag(), model),
            interview=InterviewSystem(model),
            assignments=AssignmentManager(model, ASSIGNMENTS_DIR),
            code=CodeAssistant(model),
            research=DeepResearchEngine(model),
        )
    return st.session_state.core


def reset_core():
    shared_model.clear()
    st.session_state.pop("core", None)


def save_upload(uploaded) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe = "".join(ch for ch in os.path.basename(uploaded.name) if ch.isalnum() or ch in "._-")[:80] or "file"
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{safe}")
    with open(path, "wb") as handle:
        handle.write(uploaded.getbuffer())
    return path


def log_feedback(record: dict):
    """Thumbs up/down from the chat. Doubles as human labels for the eval suite."""
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def stream_into(generator, sink: list):
    """Pass chunks through to st.write_stream while keeping a copy."""
    for piece in generator:
        if piece:
            text = piece if isinstance(piece, str) else str(piece)
            sink.append(text)
            yield text
