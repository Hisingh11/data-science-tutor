import re
from typing import Dict

class CodeAssistant:
    def __init__(self, model_manager):
        self.model = model_manager
    
    def generate_code(self, problem_description: str, language: str = "python") -> Dict:
        prompt = (
            f"Write advanced, production-style {language} for this request.\n\n"
            f"{problem_description}\n\n"
            "If source text or a file excerpt is included, treat it as the spec.\n"
            "Include a short plan, then complete code with types or clear names, "
            "error handling, and a realistic usage example. Do not leave placeholders."
        )
        response = self.model.generate_code(prompt, language)
        code = self._extract_code(response)
        
        # If no code found, try to generate a simpler version
        if not code or len(code) < 10:
            fallback_prompt = f"Write a simple {language} function for: {problem_description}. Only output the code, no explanation."
            response = self.model.generate(fallback_prompt, "code", 0.3)
            code = self._extract_code(response)
        
        return {
            "full_response": response,
            "code": code if code else "# Code generation failed. Please try a simpler request.",
            "language": language
        }
    
    def check_code(self, code: str, language: str = "python") -> str:
        return self.model.check_code_errors(code, language)

    def _extract_code(self, response: str) -> str:
        # Look for code blocks
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            if match[0].lower() in ['python', 'py', '']:
                return match[1].strip()
        
        # If no code block, try to find python-like code
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if '```' in line:
                in_code = not in_code
                continue
            if in_code or (line.strip().startswith(('def ', 'class ', 'import ', 'from '))):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # Return the whole response if it looks like code
        if 'def ' in response or 'import ' in response:
            return response
        
        return ""