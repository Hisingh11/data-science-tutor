import base64
import os

from utils.model_manager import get_groq_api_key

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None

# The configured model is tried first. If Groq reports it as unknown or
# decommissioned, the next vision-capable model is used instead.
VISION_MODELS = list(dict.fromkeys(
    model for model in (
        os.getenv("GROQ_VISION_MODEL", "").strip(),
        "qwen/qwen3.8-27b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
    ) if model
))
VISION_MODEL = VISION_MODELS[0]
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # Groq rejects base64 images above ~4 MB


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
    if not api_key or Groq is None:
        return "Image analysis error: GROQ_API_KEY is not configured."
    if os.path.getsize(image_path) > MAX_IMAGE_BYTES:
        return "Image analysis error: the image is larger than 4 MB. Resize it and try again."
    client = Groq(api_key=api_key)
    payload = f"data:{_mime_type(image_path)};base64,{encode_image(image_path)}"
    last_error = ""
    for model in VISION_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": payload}},
                    ],
                }],
                temperature=0.2,
                max_tokens=800,
            )
            text = (getattr(completion.choices[0].message, "content", None) or "").strip()
            return text or "No description returned."
        except Exception as exc:
            last_error = str(exc)
            lowered = last_error.lower()
            if "model" in lowered and ("not found" in lowered or "decommission" in lowered
                                       or "does not exist" in lowered or "not support" in lowered):
                continue
            break
    return f"Image analysis error: {last_error[:300]}"
