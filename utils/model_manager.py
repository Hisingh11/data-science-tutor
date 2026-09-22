import os
from dotenv import load_dotenv
from groq import Groq

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

TUTOR_SYSTEM = (
    "You are Data Scientist BOT, a tutor for data science, machine learning, "
    "statistics, Python, generative AI, and agentic AI. Teach clearly with "
    "examples and correct code when it helps. If you are unsure, say so. "
    "Do not invent citations or library APIs."
)


def get_groq_api_key() -> str:
    load_dotenv(dotenv_path)
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if key:
        return key
    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return ""


class ModelManager:
    def __init__(self):
        self.api_key = get_groq_api_key()
        self.init_error = ""
        self.client = None
        if not self.api_key:
            self.init_error = "GROQ_API_KEY is missing. Add it to .env or Streamlit secrets."
        else:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as exc:
                self.init_error = str(exc)

        # llama-3.1-8b-instant and llama-3.3-70b-versatile were shut down
        # for developer accounts on 16 Aug 2026.
        self.models = {
            "fast": "openai/gpt-oss-20b",
            "reasoning": "openai/gpt-oss-120b",
            "code": "openai/gpt-oss-120b",
            "vision": "qwen/qwen3.8-27b",
        }

    def _message_text(self, response) -> str:
        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
                else:
                    parts.append(getattr(part, "text", "") or str(part))
            content = "".join(parts)
        if not str(content).strip():
            content = getattr(message, "reasoning", None) or ""
        text = str(content).strip()
        return text or "The model returned an empty response. Try again."

    def answer(self, prompt, history=None, context="", model_type="reasoning", temperature=0.7):
        if not self.client:
            return self.init_error or "API key not configured"
        messages = [{"role": "system", "content": TUTOR_SYSTEM}]
        if context:
            if context.startswith("Retrieval check:"):
                note = context
            else:
                note = "Relevant notes from the knowledge base:\n" + context
            messages.append({"role": "system", "content": note})
        for msg in (history or [])[-8:]:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content[:8000]})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.models.get(model_type, self.models["reasoning"]),
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
            )
            return self._message_text(response)
        except Exception as exc:
            return f"Error: {exc}"

    def complete(self, prompt, model_type="fast", temperature=0.0):
        if not self.client:
            return self.init_error or "API key not configured"
        try:
            response = self.client.chat.completions.create(
                model=self.models.get(model_type, self.models["fast"]),
                messages=[
                    {"role": "system", "content": "Follow the requested output format exactly."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=1200,
            )
            return self._message_text(response)
        except Exception as exc:
            return f"Error: {exc}"

    def generate(self, prompt, model_type="reasoning", temperature=0.7):
        return self.answer(prompt, model_type=model_type, temperature=temperature)

    def generate_with_context(self, prompt, context, model_type="reasoning", temperature=0.7):
        return self.answer(prompt, context=context, model_type=model_type, temperature=temperature)

    def generate_code(self, prompt, language="python"):
        if not self.client:
            return self.init_error or "API key not configured"
        system_prompt = (
            f"You are an expert {language} programmer helping a data science student. "
            "Generate clean, working code with short comments and a usage example."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.models["code"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            return self._message_text(response)
        except Exception as exc:
            return f"Code error: {exc}"

    def check_code_errors(self, code, language="python"):
        prompt = (
            f"Review this {language} code for a data science student.\n"
            f"```{language}\n{code}\n```\n"
            "List bugs, then show a corrected version."
        )
        return self.generate(prompt, "code", 0.3)


_model_instance = None


def get_model_manager():
    global _model_instance
    if _model_instance is None:
        _model_instance = ModelManager()
    return _model_instance
