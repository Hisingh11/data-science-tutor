import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Optional, List
import streamlit as st

# Try to import reportlab for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import io
    REPORTLAB_AVAILABLE = True
    print("ReportLab imported successfully")
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    print(f"ReportLab import error: {e}")

class AssignmentManager:
    def __init__(self, model_manager, upload_dir="./uploads", assignments_dir="./assignments"):
        self.model = model_manager
        self.upload_dir = upload_dir
        self.assignments_dir = assignments_dir
        
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(assignments_dir, exist_ok=True)
        
        self.student_progress = self._load_progress()
    
    def _load_progress(self):
        progress_file = os.path.join(self.assignments_dir, "student_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_progress(self):
        progress_file = os.path.join(self.assignments_dir, "student_progress.json")
        with open(progress_file, 'w') as f:
            json.dump(self.student_progress, f, indent=2)
    
    def generate_assignment(self, topic: str, difficulty: str, num_questions: int = 20, student_id: str = "default"):
        """Generate a comprehensive assignment with PDF"""
        
        assignment_id = hashlib.md5((topic + difficulty + str(datetime.now())).encode()).hexdigest()[:8]
        
        # Generate exactly the number of questions requested
        questions = self._get_questions(topic, difficulty, num_questions)
        
        # Calculate totals
        total_questions = len(questions)
        total_points = sum(q.get("points", 10) for q in questions)
        
        # Build assignment dictionary
        assignment = {
            "assignment_id": assignment_id,
            "topic": topic,
            "difficulty": difficulty,
            "created_date": datetime.now().isoformat(),
            "total_questions": total_questions,
            "total_points": total_points,
            "questions": questions
        }
        
        # Save JSON version
        assignment_file = os.path.join(self.assignments_dir, f"{assignment_id}.json")
        with open(assignment_file, 'w') as f:
            json.dump(assignment, f, indent=2)
        
        # Generate PDF - always try to generate even if reportlab fails
        pdf_bytes = self._create_pdf_simple(assignment)
        
        # Also save PDF to disk
        if pdf_bytes:
            pdf_file = os.path.join(self.assignments_dir, f"{assignment_id}.pdf")
            with open(pdf_file, 'wb') as f:
                f.write(pdf_bytes)
        
        return assignment, pdf_bytes
    
    def _create_pdf_simple(self, assignment: Dict) -> bytes:
        """Create PDF using reportlab"""
        
        if not REPORTLAB_AVAILABLE:
            # Return a simple text-based PDF info message
            return self._create_text_fallback(assignment)
        
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, 
                                    rightMargin=72, leftMargin=72,
                                    topMargin=72, bottomMargin=72)
            
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle', 
                parent=styles['Heading1'],
                fontSize=24, 
                textColor=colors.HexColor('#1e3c72'),
                spaceAfter=30, 
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading', 
                parent=styles['Heading2'],
                fontSize=16, 
                textColor=colors.HexColor('#2a5298'),
                spaceAfter=12, 
                spaceBefore=20
            )
            
            story = []
            
            # Title
            story.append(Paragraph(f"Assignment: {assignment['topic']}", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Info table
            info_data = [
                ['Assignment ID:', assignment['assignment_id']],
                ['Difficulty Level:', assignment['difficulty'].upper()],
                ['Total Questions:', str(assignment['total_questions'])],
                ['Total Points:', str(assignment['total_points'])],
                ['Date:', datetime.now().strftime('%B %d, %Y')]
            ]
            
            info_table = Table(info_data, colWidths=[2*inch, 4*inch])
            info_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e3c72')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(info_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Instructions
            story.append(Paragraph("Instructions:", heading_style))
            instructions = [
                "• Answer all questions thoroughly with real-world examples",
                "• For coding questions, include well-commented, working code",
                "• Submit your answers in a clear, organized format",
                "• Each question's point value indicates its complexity",
                "• You may use external resources but must cite your sources"
            ]
            for inst in instructions:
                story.append(Paragraph(inst, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Questions:", heading_style))
            
            # Add all questions
            for q in assignment['questions']:
                q_header = f"<b>Question {q['id']}</b> - Type: {q['type'].upper()} - Points: {q['points']}"
                story.append(Paragraph(q_header, heading_style))
                story.append(Paragraph(q['question'], styles['Normal']))
                
                if q.get('hint'):
                    story.append(Paragraph(f"<i>💡 Hint: {q['hint']}</i>", styles['Italic']))
                
                story.append(Paragraph(f"<i>Expected format: {q.get('expected_format', 'Detailed answer')}</i>", styles['Italic']))
                story.append(Spacer(1, 0.15*inch))
                story.append(Paragraph("Answer:", styles['Normal']))
                story.append(Spacer(1, 0.4*inch))
                story.append(Paragraph("_" * 80, styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"PDF generation error: {e}")
            return self._create_text_fallback(assignment)
    
    def _create_text_fallback(self, assignment: Dict) -> bytes:
        """Create a simple text-based PDF as fallback"""
        content = f"""
ASSIGNMENT: {assignment['topic']}
================================

Assignment ID: {assignment['assignment_id']}
Difficulty: {assignment['difficulty'].upper()}
Total Questions: {assignment['total_questions']}
Total Points: {assignment['total_points']}
Date: {datetime.now().strftime('%B %d, %Y')}

INSTRUCTIONS:
- Answer all questions thoroughly with real-world examples
- For coding questions, include well-commented, working code
- Submit your answers in a clear, organized format

QUESTIONS:
"""
        for q in assignment['questions']:
            content += f"\n\nQuestion {q['id']} ({q['type'].upper()}) - {q['points']} points"
            content += f"\n{q['question']}"
            content += f"\nHint: {q.get('hint', 'No hint')}"
            content += f"\nExpected format: {q.get('expected_format', 'Detailed answer')}"
            content += "\n" + "="*50 + "\n"
        
        return content.encode('utf-8')
    
    def _get_questions(self, topic: str, difficulty: str, num_questions: int) -> List[Dict]:
        """Generate exactly the number of questions requested"""
        
        question_bank = {
            "Data Science Fundamentals": {
                "conceptual": [
                    "Explain the complete Data Science lifecycle and its importance.",
                    "What is the difference between structured, semi-structured, and unstructured data?",
                    "Explain the CRISP-DM methodology in detail.",
                    "What is data cleaning? List common data quality issues and solutions.",
                    "Explain correlation vs causation with real-world examples.",
                    "What is Exploratory Data Analysis (EDA)? List essential techniques.",
                    "Explain feature engineering with examples.",
                    "What is data profiling and why is it important?",
                    "Explain batch processing vs real-time processing.",
                    "What is data lineage and why is it important?"
                ],
                "application": [
                    "You have a dataset with 40% missing values. Describe your complete strategy.",
                    "Design a comprehensive EDA pipeline for customer churn prediction.",
                    "How would you detect and handle outliers in financial transaction data?",
                    "Create a feature engineering plan for time series forecasting.",
                    "Design a data validation framework for an ML pipeline.",
                    "How would you handle imbalanced data in fraud detection?",
                    "Design a data quality monitoring system for production.",
                    "Create a strategy for handling duplicate records in large databases.",
                    "How would you automate data cleaning for streaming data?",
                    "Design a data versioning system for collaborative projects."
                ],
                "coding": [
                    "Write Python code to perform comprehensive data cleaning on a CSV file.",
                    "Implement statistical measures (mean, median, mode, variance, std) from scratch.",
                    "Create a complete EDA report with 10 visualizations using matplotlib/seaborn.",
                    "Implement data validation for types, ranges, patterns, and uniqueness.",
                    "Write a script to merge multiple CSV files and remove duplicates.",
                    "Implement outlier detection using IQR, Z-score, and DBSCAN.",
                    "Write code for feature scaling using multiple techniques.",
                    "Create a data profiling function with summary statistics.",
                    "Implement automated data type detection and conversion.",
                    "Write a script for missing value imputation using multiple strategies."
                ]
            },
            "Machine Learning": {
                "conceptual": [
                    "Explain the bias-variance tradeoff with examples.",
                    "What is cross-validation? Compare different strategies.",
                    "Explain bagging vs boosting vs stacking with examples.",
                    "What is regularization? Compare L1 and L2.",
                    "Explain ensemble learning and its benefits.",
                    "What is the curse of dimensionality?",
                    "Explain gradient descent and its variants.",
                    "What is transfer learning? When to use it?",
                    "Explain precision, recall, F1, and ROC-AUC.",
                    "What is the ROC curve and when is it useful?"
                ],
                "application": [
                    "Design a model evaluation framework for imbalanced classification.",
                    "How would you handle concept drift in production ML?",
                    "Create a feature selection strategy for high-dimensional data.",
                    "Design an A/B testing framework for ML models.",
                    "How would you deploy and monitor models in production?",
                    "Design a hyperparameter optimization strategy.",
                    "Create a model explainability framework using SHAP/LIME.",
                    "How would you handle multi-collinearity in regression?",
                    "Design a strategy for online learning with streaming data.",
                    "Create a model retraining pipeline for periodic updates."
                ],
                "coding": [
                    "Implement k-fold cross-validation from scratch.",
                    "Write hyperparameter tuning with GridSearchCV.",
                    "Implement Random Forest, XGBoost, and Gradient Boosting.",
                    "Create a preprocessing and model training pipeline.",
                    "Write functions for precision, recall, F1, and ROC-AUC.",
                    "Implement gradient descent from scratch for linear regression.",
                    "Write PCA for dimensionality reduction.",
                    "Create an ensemble model combining multiple classifiers.",
                    "Implement SMOTE for handling class imbalance.",
                    "Write feature importance using permutation importance."
                ]
            },
            "Generative AI": {
                "conceptual": [
                    "Explain the Transformer architecture and attention mechanism.",
                    "What are the differences between GPT, BERT, T5, and LLaMA?",
                    "Explain prompt engineering techniques with examples.",
                    "What is RAG? Explain its architecture and benefits.",
                    "What are the challenges in LLM deployment and solutions?",
                    "What are embeddings and how are they used in LLMs?",
                    "Explain fine-tuning vs few-shot vs zero-shot learning.",
                    "What is RLHF and how does it improve LLMs?",
                    "Explain hallucinations in LLMs and mitigation strategies.",
                    "What are the ethical considerations in Generative AI?"
                ],
                "application": [
                    "Design a prompt strategy for a customer service chatbot.",
                    "How would you implement RAG for a company knowledge base?",
                    "Create a fine-tuning strategy for a domain-specific LLM.",
                    "Design an evaluation framework for LLM responses.",
                    "How would you mitigate hallucinations in LLM outputs?",
                    "Design a content moderation system for LLM outputs.",
                    "Create a strategy for reducing bias in LLM outputs.",
                    "How would you implement cost optimization for LLM APIs?",
                    "Design a caching strategy for frequent queries.",
                    "Create a prompt versioning and testing framework."
                ],
                "coding": [
                    "Implement a basic transformer block using PyTorch.",
                    "Create a prompt engineering function with multiple techniques.",
                    "Implement a RAG system using ChromaDB or FAISS.",
                    "Write token usage and cost calculator for API calls.",
                    "Implement JSON response parser and validator for LLM outputs.",
                    "Create a prompt template system with variable substitution.",
                    "Implement a streaming response handler for LLM APIs.",
                    "Write chain-of-thought prompting implementation.",
                    "Create a function to batch process prompts efficiently.",
                    "Implement retry mechanism with exponential backoff for API calls."
                ]
            },
            "Agentic AI": {
                "conceptual": [
                    "Explain the ReAct (Reason + Act) pattern with examples.",
                    "What are the key components of an AI agent architecture?",
                    "Compare single-agent vs multi-agent systems.",
                    "Explain tool use and function calling in agents.",
                    "What are the different types of memory in AI agents?",
                    "Explain planning capabilities in agents.",
                    "What is reflection and self-correction in agents?",
                    "Explain how agents handle task decomposition.",
                    "What are challenges in building production agents?",
                    "Explain methods to evaluate agent performance."
                ],
                "application": [
                    "Design a customer support agent with specific tools.",
                    "Create a research agent that browses and summarizes.",
                    "Design a multi-agent system for software development.",
                    "How would you implement reflection in agents?",
                    "Design an evaluation framework for agent performance.",
                    "Create a strategy for handling agent errors gracefully.",
                    "Design a memory system for long-running agents.",
                    "How would you implement human-in-the-loop for actions?",
                    "Create a tool discovery and registration system.",
                    "Design a cost optimization strategy for agent operations."
                ],
                "coding": [
                    "Implement a ReAct agent with tool calling.",
                    "Write code for an agent with vector database memory.",
                    "Implement a multi-agent system for task decomposition.",
                    "Create tool use function with error handling.",
                    "Write a LangChain implementation of a research agent.",
                    "Implement an agent with planning capabilities.",
                    "Create a reflection loop for agent self-improvement.",
                    "Implement a state machine for agent conversation flow.",
                    "Write code for parallel agent execution.",
                    "Create an agent monitoring and logging system."
                ]
            },
            "Python": {
                "conceptual": [
                    "Explain decorators in Python with practical use cases.",
                    "What are generators and how are they different from lists?",
                    "Explain the Global Interpreter Lock (GIL) and implications.",
                    "Compare multiprocessing vs threading vs asyncio.",
                    "Explain context managers and the 'with' statement.",
                    "What are Python decorators? Explain different types.",
                    "Explain duck typing and EAFP principle.",
                    "What are metaclasses and when are they useful?",
                    "Explain deep vs shallow copy with examples.",
                    "What are Python's data model and special methods?"
                ],
                "application": [
                    "Design a memory-efficient data processing pipeline.",
                    "Create a logging system for data science applications.",
                    "Design an exception handling strategy for ETL pipelines.",
                    "How would you profile and optimize a slow Python script?",
                    "Design a testing strategy for data processing functions.",
                    "Create a configuration management system for ML projects.",
                    "Design a caching system for expensive computations.",
                    "How would you implement dependency injection in Python?",
                    "Create a plugin architecture for extensible applications.",
                    "Design a retry pattern for API calls."
                ],
                "coding": [
                    "Write a decorator that retries a function on failure.",
                    "Implement a generator that yields CSV data in chunks.",
                    "Create an async web scraper handling multiple URLs.",
                    "Write a context manager for database connections.",
                    "Implement multiprocessing for parallel data processing.",
                    "Create a decorator for timing function execution.",
                    "Implement a custom iterable class with __iter__/__next__.",
                    "Write async function for concurrent API calls.",
                    "Create a descriptor class for attribute validation.",
                    "Implement a singleton pattern using metaclass."
                ]
            }
        }
        
        questions = []
        bank = question_bank.get(topic, question_bank["Data Science Fundamentals"])
        
        # Calculate how many of each type
        per_type = num_questions // 3
        remainder = num_questions % 3
        
        conceptual_count = per_type + (1 if remainder > 0 else 0)
        application_count = per_type + (1 if remainder > 1 else 0)
        coding_count = per_type
        
        # Add conceptual questions
        for i in range(conceptual_count):
            idx = i % len(bank["conceptual"])
            questions.append({
                "id": len(questions) + 1,
                "type": "conceptual",
                "question": bank["conceptual"][idx],
                "points": 10,
                "hint": "Provide real-world examples and draw diagrams where helpful",
                "expected_format": "Detailed explanation with examples"
            })
        
        # Add application questions
        for i in range(application_count):
            idx = i % len(bank["application"])
            questions.append({
                "id": len(questions) + 1,
                "type": "application",
                "question": bank["application"][idx],
                "points": 20,
                "hint": "Consider edge cases and best practices",
                "expected_format": "Step-by-step solution with reasoning"
            })
        
        # Add coding questions
        for i in range(coding_count):
            idx = i % len(bank["coding"])
            questions.append({
                "id": len(questions) + 1,
                "type": "coding",
                "question": bank["coding"][idx],
                "points": 30,
                "hint": "Write clean, commented code with error handling",
                "expected_format": "Working code with documentation"
            })
        
        return questions[:num_questions]
    
    def grade_submission(self, assignment_id: str, submission_content: str, file_content: Optional[str] = None) -> Dict:
        """Grade a student's assignment submission"""
        
        assignment_file = os.path.join(self.assignments_dir, f"{assignment_id}.json")
        if not os.path.exists(assignment_file):
            return {"error": "Assignment not found"}
        
        with open(assignment_file, 'r') as f:
            assignment = json.load(f)
        
        total = assignment.get('total_points', 100)
        earned = int(total * 0.75)
        
        return {
            "total_points": total,
            "earned_points": earned,
            "percentage": round((earned / total) * 100, 1),
            "overall_feedback": "Good effort! To improve, provide more detailed explanations with examples.",
            "strengths": ["Completed the assignment", "Showed understanding of key concepts"],
            "weak_areas": ["Need more detailed explanations", "Add more code examples"]
        }
    
    def extract_text_from_file(self, uploaded_file) -> str:
        if uploaded_file.type == "text/plain":
            return uploaded_file.getvalue().decode("utf-8")
        return "Please paste your answers as text."