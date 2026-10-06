"""Intent routing for the tutor chat box.

The old router matched keywords anywhere in the message, so ordinary questions
such as "what is an assignment operator in Python?" generated an assignment and
"what are common ML interview questions?" started a mock interview. Intents are
now matched on imperative phrasing, and the function is pure so it is covered
by the routing eval (``evals/datasets/routing.jsonl``).
"""

import re
from typing import Dict

INTERVIEW_TOPICS = {
    "data science": "Data Science Fundamentals",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "generative": "Generative AI",
    "gen ai": "Generative AI",
    "genai": "Generative AI",
    "llm": "Generative AI",
    "agent": "Agentic AI",
    "agentic": "Agentic AI",
    "python": "Python & Coding",
    "coding": "Python & Coding",
}
ASSIGNMENT_TOPICS = {**INTERVIEW_TOPICS, "python": "Python", "coding": "Python"}

DIFFICULTIES = ("beginner", "intermediate", "advanced")

_CODE_NOUN = r"(code|function|script|class|program|snippet|query|implementation|decorator|generator)"


def pick_topic(text: str, table: Dict[str, str], default: str) -> str:
    lowered = (text or "").lower()
    for key, topic in table.items():
        if re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", lowered):
            return topic
    return default


def pick_difficulty(text: str, default: str = "intermediate") -> str:
    lowered = (text or "").lower()
    aliases = {"easy": "beginner", "basic": "beginner", "medium": "intermediate",
               "hard": "advanced", "expert": "advanced"}
    for level in DIFFICULTIES:
        if level in lowered:
            return level
    for word, level in aliases.items():
        if re.search(rf"\b{word}\b", lowered):
            return level
    return default


def pick_language(text: str) -> str:
    lowered = (text or "").lower()
    if re.search(r"\bsql\b|\bquery\b.*\btable\b", lowered):
        return "sql"
    if re.search(r"(?<![a-z'’])r(?![a-z'’])", lowered) and "python" not in lowered:
        return "r"
    return "python"


def classify_intent(text: str) -> str:
    """Return one of: tutor, interview, assignment, grade, research, fact_check,
    code_review, code_generate."""
    raw = (text or "").strip()
    # Drop polite lead-ins so "can you research X" routes like "research X".
    lowered = re.sub(r"^(please |can you |could you |would you |hey,? |pls )+(please )?", "", raw.lower())
    if not lowered:
        return "tutor"

    if re.match(r"(fact[- ]?check|verify( this| the claim)?|is it true that)\b", lowered):
        return "fact_check"
    if re.match(r"(research|look up|deep dive( into| on)?|find sources)\b", lowered):
        return "research"
    if re.match(r"grade\b", lowered):
        return "grade"
    if re.search(r"\b(interview|quiz|test) me\b", lowered) or re.match(
        r"(start |begin |run |do )?(a |an )?(mock |practice )?interview\b", lowered
    ):
        return "interview"
    if re.match(r"(assignment|homework|worksheet)\b", lowered) or re.search(
        r"\b(give|create|generate|make|build|set|prepare)\b[^.?!]{0,30}\b(assignment|practice (questions|set|problems)|worksheet|homework)\b",
        lowered,
    ):
        return "assignment"
    if "```" in raw or re.match(r"(review|debug|fix|refactor|optimi[sz]e)\b[^.?!]{0,40}\b(code|function|script|this|bug)", lowered):
        return "code_review"
    # Questions ("why/what/how ...") are explanations, not code requests; the
    # tutor still includes code when it helps.
    question = re.match(r"(why|what|how|when|where|which|who|is|are|does|do|should)\b", lowered)
    if not question and re.search(rf"\b(write|generate|implement|create|build|code up)\b[^.?!]{{0,40}}\b{_CODE_NOUN}\b", lowered) or re.match(
        r"(write (python|sql|r)\b|implement\b|code up\b)", lowered
    ):
        return "code_generate"
    return "tutor"
