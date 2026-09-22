import base64
import os

from groq import Groq

from utils.model_manager import get_groq_api_key

VISION_MODEL = "qwen/qwen3.8-27b"


def encode_image(image_path):
    with open(image_path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def _mime_type(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


def analyze_image(image_path, prompt="Describe this image in detail"):
    api_key = get_groq_api_key()
    if not api_key:
        return "Image analysis error: GROQ_API_KEY is not configured."
    client = Groq(api_key=api_key)
    mime = _mime_type(image_path)
    try:
        completion = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encode_image(image_path)}"},
                        },
                    ],
                }
            ],
            temperature=0.4,
            max_tokens=800,
        )
        message = completion.choices[0].message
        text = (getattr(message, "content", None) or getattr(message, "reasoning", None) or "").strip()
        return text or "No description returned."
    except Exception as exc:
        return f"Image analysis error: {exc}"
