import json
import random
from typing import List, Dict
from datetime import datetime

class InterviewSystem:
    def __init__(self, model_manager):
        self.model = model_manager
        self.interview_history = []
        self.current_topic = None
        self.current_difficulty = None
        self.score = 0
        self.questions_asked = 0
        self.questions = []
    
    QUESTION_BANK = {
        "Data Science Fundamentals": {
            "beginner": [
                "What is the difference between supervised and unsupervised learning? Give examples.",
                "Explain the bias-variance tradeoff in machine learning.",
                "What is cross-validation and why is it important?",
                "Describe the steps in a typical data science project lifecycle.",
                "What is overfitting and how can you prevent it?"
            ],
            "intermediate": [
                "Explain the difference between L1 and L2 regularization.",
                "Describe how Gradient Boosting works.",
                "How do you handle imbalanced datasets?",
                "What is the curse of dimensionality?",
                "Explain precision, recall, and F1 score."
            ],
            "advanced": [
                "Derive the gradient descent update rule for logistic regression.",
                "Explain the mathematical intuition behind the kernel trick in SVMs.",
                "Compare Bagging, Boosting, and Stacking ensemble methods.",
                "How would you detect and mitigate data leakage?",
                "Explain the Expectation-Maximization algorithm."
            ]
        },
        "Machine Learning": {
            "beginner": [
                "What is the difference between classification and regression?",
                "Explain how a decision tree makes decisions.",
                "What is the purpose of a confusion matrix?",
                "How does K-Nearest Neighbors work?",
                "What is feature scaling and why is it important?"
            ],
            "intermediate": [
                "Explain how Random Forest reduces overfitting.",
                "Describe how PCA works for dimensionality reduction.",
                "What is the difference between batch and stochastic gradient descent?",
                "How would you handle missing values in a dataset?",
                "Explain the concept of ensemble learning."
            ],
            "advanced": [
                "Explain the mathematics behind Attention mechanisms.",
                "Derive the backpropagation algorithm for a neural network.",
                "What is the vanishing gradient problem?",
                "Explain the Vapnik-Chervonenkis (VC) dimension.",
                "Compare Adam, RMSprop, and Adagrad optimizers."
            ]
        },
        "Generative AI": {
            "beginner": [
                "What are Large Language Models and how do they work?",
                "Explain prompt engineering with examples.",
                "What is the difference between zero-shot and few-shot learning?",
                "How does text generation work in GPT models?",
                "What are embeddings and why are they important?"
            ],
            "intermediate": [
                "Explain the Transformer architecture and self-attention.",
                "What is RAG and when would you use it?",
                "Describe fine-tuning vs prompt engineering.",
                "How does RLHF work?",
                "What are hallucinations in LLMs and how to mitigate them?"
            ],
            "advanced": [
                "Explain the mathematical formulation of attention.",
                "Describe how diffusion models work.",
                "Compare encoder-only, decoder-only, and encoder-decoder architectures.",
                "Explain Mixture of Experts (MoE).",
                "What are the challenges in scaling LLMs?"
            ]
        },
        "Agentic AI": {
            "beginner": [
                "What is an AI agent? How is it different from a traditional LLM?",
                "Explain the ReAct (Reason + Act) pattern.",
                "What are tools in AI agents? Give examples.",
                "How does memory work in AI agents?",
                "What is the difference between single and multi-agent systems?"
            ],
            "intermediate": [
                "Describe the components of a complete agent architecture.",
                "How does planning work in agentic systems?",
                "What is chain-of-thought reasoning?",
                "Explain tool use and function calling.",
                "How do you implement reflection in agents?"
            ],
            "advanced": [
                "Design a multi-agent system for automated research.",
                "How to implement long-term memory using vector databases?",
                "What are the challenges in building production agents?",
                "How do you evaluate agent performance?",
                "Explain hierarchical agent architectures."
            ]
        },
        "Python & Coding": {
            "beginner": [
                "Write a function to find factorial using recursion.",
                "How do you handle exceptions in Python?",
                "Explain list comprehensions with examples.",
                "What are decorators? Write a simple decorator.",
                "Write a function to check if a string is a palindrome."
            ],
            "intermediate": [
                "Implement a stack class with push, pop, and peek.",
                "Write a generator that yields Fibonacci numbers.",
                "Explain multiprocessing vs threading.",
                "Write a decorator that measures execution time.",
                "Write a function to find duplicates in a list."
            ],
            "advanced": [
                "Implement a context manager for file handling.",
                "Write a metaclass that adds logging to methods.",
                "Explain Python's GIL and its implications.",
                "Implement an async web scraper using asyncio.",
                "Write a memory-efficient CSV processor using generators."
            ]
        }
    }
    
    def start_interview(self, topic: str, difficulty: str):
        """Start a new interview session"""
        self.current_topic = topic
        self.current_difficulty = difficulty
        self.interview_history = []
        self.score = 0
        self.questions_asked = 0
        
        if topic in self.QUESTION_BANK and difficulty in self.QUESTION_BANK[topic]:
            self.questions = self.QUESTION_BANK[topic][difficulty].copy()
            random.shuffle(self.questions)
            return True
        return False
    
    def get_next_question(self):
        """Get the next interview question. Safe to call on every rerun."""
        if not self.questions:
            return None
        if self.questions_asked < len(self.interview_history):
            current = self.interview_history[self.questions_asked]
            if current["user_answer"] is None:
                return current["question"]
        if self.questions_asked < len(self.questions):
            question = self.questions[self.questions_asked]
            self.interview_history.append({
                "question": question,
                "user_answer": None,
                "ai_feedback": None,
                "score": None
            })
            return question
        return None
    
    def evaluate_answer(self, user_answer: str):
        """Evaluate user's answer and provide feedback"""
        if self.questions_asked >= len(self.interview_history):
            return None
        
        current = self.interview_history[self.questions_asked]
        current["user_answer"] = user_answer
        
        eval_prompt = f"""Evaluate this data science interview answer.
Topic: {self.current_topic}
Difficulty: {self.current_difficulty}
Question: {current['question']}
Candidate's answer: {user_answer}

score is an integer from 0 to 10.
strengths and improvements are short lists.
model_answer is a strong sample answer.
feedback is one constructive paragraph."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer"},
                "strengths": {"type": "array", "items": {"type": "string"}},
                "improvements": {"type": "array", "items": {"type": "string"}},
                "model_answer": {"type": "string"},
                "feedback": {"type": "string"},
            },
            "required": ["score", "strengths", "improvements", "model_answer", "feedback"],
        }
        evaluation = self.model.complete_json(eval_prompt, schema, "interview_grade", "reasoning", 0.2)
        if not evaluation:
            evaluation = self._default_evaluation()
        try:
            evaluation["score"] = max(0, min(10, int(evaluation.get("score", 5))))
        except (TypeError, ValueError):
            evaluation["score"] = 5
        
        current["ai_feedback"] = evaluation.get("feedback", "Good attempt")
        current["model_answer"] = evaluation.get("model_answer", current['question'])
        current["score"] = evaluation.get("score", 5)
        
        self.score += current["score"]
        self.questions_asked += 1
        
        return evaluation
    
    def _default_evaluation(self):
        return {
            "score": 5,
            "strengths": ["Attempted the answer"],
            "improvements": ["Provide more details"],
            "model_answer": "Provide a detailed answer with examples",
            "feedback": "Good attempt. Review the concepts and try to provide more examples."
        }
    
    def get_summary(self):
        """Get interview summary"""
        if self.questions_asked == 0:
            return "No questions answered yet.", 0, 0
        
        avg_score = self.score / self.questions_asked
        max_score = self.questions_asked * 10
        percentage = (self.score / max_score) * 100
        
        summary = f"Interview completed! Score: {self.score}/{max_score} ({percentage:.1f}%)"
        
        return summary, avg_score, percentage