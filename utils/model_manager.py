import json
import os
from dotenv import load_dotenv
from groq import Groq

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")


def get_groq_api_key():
    load_dotenv(dotenv_path)
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if key:
        return key
    try:
        import streamlit as st

        return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        return ""


def trim_history(history, total_chars=4000, message_chars=1000):
    kept = []
    used = 0
    for message in reversed(history or []):
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        content = content[:message_chars]
        if used + len(content) > total_chars:
            break
        kept.append({"role": role, "content": content})
        used += len(content)
    return list(reversed(kept))

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

        self.models = {
            "fast": "openai/gpt-oss-20b",
            "reasoning": "openai/gpt-oss-120b",
            "code": "openai/gpt-oss-120b",
        }

    def _chat_messages(self, prompt, history=None, context=""):
        messages = [{
            "role": "system",
            "content": (
                "You are an expert data science tutor. Teach clearly, use examples when helpful, "
                "and say when you are unsure. Do not invent citations or library APIs."
            ),
        }]
        if context:
            messages.append({
                "role": "system",
                "content": f"Relevant notes for this answer:\n{context[:10000]}",
            })
        messages.extend(trim_history(history))
        messages.append({"role": "user", "content": prompt})
        return messages

    def _message_text(self, response):
        message = response.choices[0].message
        return (getattr(message, "content", None) or getattr(message, "reasoning", None) or "").strip()

    def _error_text(self, exc):
        return f"Error: {exc}"

    def answer(self, prompt, history=None, context="", model_type="reasoning", temperature=0.7):
        if not self.client:
            return self.init_error or "API key not configured"
        try:
            response = self.client.chat.completions.create(
                model=self.models.get(model_type, self.models["reasoning"]),
                messages=self._chat_messages(prompt, history, context),
                temperature=temperature,
                max_tokens=4096,
            )
            return self._message_text(response) or "The model returned an empty response. Try again."
        except Exception as exc:
            return self._error_text(exc)

    def complete_json(self, prompt, schema, name="result", model_type="reasoning", temperature=0.2):
        if not self.client:
            return None
        model = self.models.get(model_type, self.models["reasoning"])
        formats = [
            {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
            {"type": "json_object"},
        ]
        for response_format in formats:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Return only JSON matching the requested schema."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=2048,
                    response_format=response_format,
                )
                parsed = json.loads(self._message_text(response))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    def stream_answer(self, prompt, history=None, context="", model_type="reasoning", temperature=0.6):
        if not self.client:
            yield self.init_error or "API key not configured"
            return
        try:
            stream = self.client.chat.completions.create(
                model=self.models.get(model_type, self.models["reasoning"]),
                messages=self._chat_messages(prompt, history, context),
                temperature=temperature,
                max_tokens=4096,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices:
                    text = getattr(chunk.choices[0].delta, "content", None)
                    if text:
                        yield text
        except Exception as exc:
            yield self._error_text(exc)

    def stream_code(self, prompt, language="python"):
        code_prompt = f"Write clean, working {language} code for this request:\n\n{prompt}"
        yield from self.stream_answer(code_prompt, model_type="code", temperature=0.3)

    def generate(self, prompt, model_type="reasoning", temperature=0.7):
        return self.answer(prompt, model_type=model_type, temperature=temperature)

_model_instance = None

def get_model_manager():
    global _model_instance
    if _model_instance is None:
        _model_instance = ModelManager()
    return _model_instance