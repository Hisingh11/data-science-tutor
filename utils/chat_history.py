import json
import os
from typing import Dict, List

HISTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chat_history.json")

# Keep the file small. The model only replays the last 16 turns anyway,
# and _chat_messages truncates each one at 8000 characters.
MAX_MESSAGES = 200
MAX_MODEL_CONTENT = 8000


def _clean(message: Dict) -> Dict:
    role = message.get("role")
    if role not in ("user", "assistant"):
        return {}
    content = str(message.get("content") or "")
    if not content.strip():
        return {}
    model_content = str(message.get("model_content") or content)
    attachments = [str(path) for path in (message.get("attachments") or [])]
    return {
        "role": role,
        "content": content,
        "model_content": model_content[:MAX_MODEL_CONTENT],
        "attachments": attachments,
    }


def load_messages() -> List[Dict]:
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return []
    if not isinstance(saved, list):
        return []
    messages = [_clean(item) for item in saved if isinstance(item, dict)]
    return [item for item in messages if item][-MAX_MESSAGES:]


def save_messages(messages: List[Dict]) -> None:
    cleaned = [_clean(item) for item in messages]
    cleaned = [item for item in cleaned if item][-MAX_MESSAGES:]
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    temp_path = HISTORY_PATH + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(cleaned, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, HISTORY_PATH)
    except OSError:
        # Never let a disk problem break the chat.
        try:
            os.remove(temp_path)
        except OSError:
            pass


def clear_messages() -> None:
    try:
        os.remove(HISTORY_PATH)
    except OSError:
        pass
