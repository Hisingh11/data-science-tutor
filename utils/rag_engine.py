import os
import json
from typing import List, Dict
import streamlit as st

class RAGEngine:
    def __init__(self, persist_directory="./data/knowledge_base"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.documents = []
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to in‑memory knowledge base"""
        for doc in documents:
            self.documents.append({
                "text": doc['text'],
                "metadata": doc.get('metadata', {})
            })
        # No success message – keep UI clean
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Simple keyword‑based search"""
        query_lower = query.lower()
        results = []
        for doc in self.documents:
            text_lower = doc['text'].lower()
            # Count matching words
            score = sum(1 for word in query_lower.split() if word in text_lower)
            if score > 0:
                results.append({
                    'text': doc['text'],
                    'metadata': doc['metadata'],
                    'relevance': score / max(len(query_lower.split()), 1)
                })
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:n_results]
    
    def load_initial_knowledge(self):
        """Load initial knowledge base"""
        knowledge_docs = [
            {"text": "Data Science Lifecycle: Problem Definition, Data Collection, Data Cleaning, EDA, Feature Engineering, Model Selection, Training, Evaluation, Deployment, Monitoring", "metadata": {"topic": "Data Science"}},
            {"text": "Machine Learning: Supervised (Regression, Classification), Unsupervised (Clustering, PCA), Evaluation Metrics (MSE, Accuracy, F1, AUC-ROC)", "metadata": {"topic": "ML"}},
            {"text": "Generative AI: LLMs, Transformers, Attention Mechanism, Prompt Engineering, RAG, Fine-tuning", "metadata": {"topic": "GenAI"}},
            {"text": "Agentic AI: AI Agents, ReAct Pattern, Planning, Memory, Tool Use, Multi-agent Systems", "metadata": {"topic": "Agentic AI"}},
            {"text": "Python for Data Science: NumPy, Pandas, Matplotlib, Scikit-learn for data manipulation and ML", "metadata": {"topic": "Python"}}
        ]
        self.add_documents(knowledge_docs)
        return len(knowledge_docs)