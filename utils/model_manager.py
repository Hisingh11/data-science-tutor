import json
import os
from dotenv import load_dotenv
from groq import Groq

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

# Groq's on-demand tier allows 8,000 tokens per minute per model, and the
# request check counts prompt tokens plus max_tokens. Everything below is
# sized so one request stays under that.
TOKEN_LIMIT = 8000
SAFETY_TOKENS = 400
CHARS_PER_TOKEN = 3.5
MAX_OUTPUT_TOKENS = 2500
MIN_OUTPUT_TOKENS = 600
HISTORY_CHARS = 6000
HISTORY_MESSAGE_CHARS = 1500
CONTEXT_CHARS = 3500


def estimate_tokens(text) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 4


def trim_history(history, total_chars=HISTORY_CHARS, message_chars=HISTORY_MESSAGE_CHARS):
    kept = []
    used = 0
    for msg in reversed(history or []):
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        content = content[:message_chars]
        if used + len(content) > total_chars:
            break
        kept.append({"role": role, "content": content})
        used += len(content)
    kept.reverse()
    return kept

TUTOR_SYSTEM = (
    "You are Data Scientist BOT, a tutor for data science, machine learning, "
    "statistics, Python, generative AI, and agentic AI. Teach clearly with "
    "examples and correct code when it helps. If you are unsure, say so. "
    "Do not invent citations or library APIs. "
    "Use the earlier messages in this conversation. Do not ask the user to repeat what they already said."
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

    def _error_text(self, exc) -> str:
        text = str(exc)
        if "Request too large" in text or "rate_limit_exceeded" in text or "429" in text:
            return (
                "Error: the Groq free tier allows about 8,000 tokens per minute for this model. "
                "Wait a minute and try again, or attach smaller files."
            )
        return f"Error: {text}"

    def _fit(self, messages, wanted_output=MAX_OUTPUT_TOKENS):
        """Drop old turns, then trim the prompt, so prompt + output fit TOKEN_LIMIT."""
        budget = TOKEN_LIMIT - SAFETY_TOKENS

        def used():
            return sum(estimate_tokens(m["content"]) for m in messages)

        while used() + MIN_OUTPUT_TOKENS > budget:
            index = next((i for i, m in enumerate(messages[:-1]) if m["role"] != "system"), None)
            if index is None:
                break
            messages.pop(index)
        over = used() + MIN_OUTPUT_TOKENS - budget
        if over > 0:
            last = messages[-1]
            keep = max(200, len(last["content"]) - int(over * CHARS_PER_TOKEN) - 100)
            last["content"] = (
                last["content"][:keep].rstrip()
                + "\n\n[Message truncated to fit the model's token limit.]"
            )
        output = min(wanted_output, budget - used())
        return messages, max(MIN_OUTPUT_TOKENS, output)

    def _chat_messages(self, prompt, history=None, context=""):
        messages = [{"role": "system", "content": TUTOR_SYSTEM}]
        if context:
            context = context[:CONTEXT_CHARS]
            if context.startswith("Retrieval check:"):
                note = context
            else:
                note = "Relevant notes from the knowledge base:\n" + context
            messages.append({"role": "system", "content": note})
        messages.extend(trim_history(history))
        messages.append({"role": "user", "content": prompt})
        return messages

    def answer(self, prompt, history=None, context="", model_type="reasoning", temperature=0.7):
        if not self.client:
            return self.init_error or "API key not configured"
        try:
            messages, max_tokens = self._fit(self._chat_messages(prompt, history, context))
            response = self.client.chat.completions.create(
                model=self.models.get(model_type, self.models["reasoning"]),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return self._message_text(response)
        except Exception as exc:
            return self._error_text(exc)

    def stream_answer(self, prompt, history=None, context="", model_type="reasoning", temperature=0.6):
        if not self.client:
            yield self.init_error or "API key not configured"
            return
        try:
            messages, max_tokens = self._fit(self._chat_messages(prompt, history, context))
            stream = self.client.chat.completions.create(
                model=self.models.get(model_type, self.models["reasoning"]),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as exc:
            yield self._error_text(exc)
            return
        saw_content = False
        reasoning = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None) or ""
            if text:
                saw_content = True
                yield text
            thought = getattr(delta, "reasoning", None) or ""
            if thought:
                reasoning.append(thought)
        if not saw_content and reasoning:
            yield "".join(reasoning)

    def complete_json(self, prompt, schema, name="result", model_type="reasoning", temperature=0.2):
        if not self.client:
            return None
        messages, max_tokens = self._fit([
            {"role": "system", "content": "Return only JSON that matches the schema."},
            {"role": "user", "content": prompt},
        ], wanted_output=2048)
        model_name = self.models.get(model_type, self.models["reasoning"])
        formats = [
            {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
            {"type": "json_object"},
        ]
        for response_format in formats:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                raw = self._message_text(response)
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    def complete(self, prompt, model_type="fast", temperature=0.0):
        if not self.client:
            return self.init_error or "API key not configured"
        try:
            messages, max_tokens = self._fit([
                {"role": "system", "content": "Follow the requested output format exactly."},
                {"role": "user", "content": prompt},
            ], wanted_output=1200)
            response = self.client.chat.completions.create(
                model=self.models.get(model_type, self.models["fast"]),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return self._message_text(response)
        except Exception as exc:
            return self._error_text(exc)

    def generate(self, prompt, model_type="reasoning", temperature=0.7):
        return self.answer(prompt, model_type=model_type, temperature=temperature)

    def generate_code(self, prompt, language="python"):
        if not self.client:
            return self.init_error or "API key not configured"
        system_prompt = (
            f"You are an expert {language} programmer helping a data science student. "
            "Generate clean, working code with short comments and a usage example."
        )
        try:
            messages, max_tokens = self._fit([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ])
            response = self.client.chat.completions.create(
                model=self.models["code"],
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return self._message_text(response)
        except Exception as exc:
            return self._error_text(exc)

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
