import base64
import os
from groq import Groq

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def analyze_image(image_path, prompt="Describe this image in detail"):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    base64_image = encode_image(image_path)
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.5,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Image analysis error: {str(e)}"