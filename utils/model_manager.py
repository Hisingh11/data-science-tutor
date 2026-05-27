import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

class ModelManager:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            st.error("GROQ_API_KEY not found")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                st.success("✅ API ready")
            except Exception as e:
                st.error(f"API error: {e}")
                self.client = None
        
        self.models = {
            "fast": "llama-3.1-8b-instant",
            "reasoning": "llama-3.3-70b-versatile",
            "code": "llama-3.3-70b-versatile"
        }
    
    def generate(self, prompt, model_type="reasoning", temperature=0.7):
        if not self.client:
            return "API key not configured"
        try:
            model_name = self.models.get(model_type, self.models["reasoning"])
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_with_context(self, prompt, context, model_type="reasoning", temperature=0.7):
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
        return self.generate(full_prompt, model_type, temperature)
    
    def generate_code(self, prompt, language="python"):
        if not self.client:
            return "API key not configured"
        system_prompt = f"You are an expert {language} programmer. Generate clean code with comments."
        try:
            response = self.client.chat.completions.create(
                model=self.models["code"],
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Code error: {str(e)}"
    
    def check_code_errors(self, code, language="python"):
        prompt = f"Analyze this {language} code:\n```{language}\n{code}\n```\nList errors and suggest fixes."
        return self.generate(prompt, "code", 0.3)

_model_instance = None

def get_model_manager():
    global _model_instance
    if _model_instance is None:
        _model_instance = ModelManager()
    return _model_instance
