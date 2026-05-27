import subprocess
import sys
import tempfile
import re
from typing import Dict

class CodeAssistant:
    def __init__(self, model_manager):
        self.model = model_manager
    
    def generate_code(self, problem_description: str, language: str = "python") -> Dict:
        prompt = f"Generate {language} code for: {problem_description}\n\nProvide clean, working code with comments and example usage."
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
    
    def check_code(self, code: str, language: str = "python") -> Dict:
        return self.model.check_code_errors(code, language)
    
    def execute_python_code(self, code: str, timeout_seconds: int = 10) -> Dict:
        result = {"success": False, "output": "", "error": "", "return_code": None}
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            process = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["return_code"] = process.returncode
            result["success"] = process.returncode == 0
        except subprocess.TimeoutExpired:
            result["error"] = f"Timeout after {timeout_seconds} seconds"
        except Exception as e:
            result["error"] = str(e)
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return result
    
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
    
    def resolve_code_error(self, code: str, error_message: str) -> str:
        prompt = f"Fix this code error:\n\nCODE:\n```python\n{code}\n```\n\nERROR:\n{error_message}\n\nProvide the fixed code."
        return self.model.generate(prompt, "code", 0.3)
    
    def analyze_image_code(self, extracted_text: str) -> Dict:
        prompt = f"Solve this coding problem:\n{extracted_text}\n\nProvide the solution with code."
        response = self.model.generate(prompt, "code", 0.4)
        code = self._extract_code(response)
        return {
            "analysis": response,
            "code": code,
            "language": "python"
        }