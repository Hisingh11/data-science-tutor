"""
RAG Engine - Simple version for Streamlit Cloud
"""

import os
import json
from typing import List, Dict
import streamlit as st

class RAGEngine:
    def __init__(self, persist_directory="./data/knowledge_base"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Simple in-memory storage for demo
        self.documents = []
        self.knowledge_base = []
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to knowledge base"""
        for doc in documents:
            self.knowledge_base.append({
                "text": doc['text'],
                "metadata": doc.get('metadata', {})
            })
        st.success(f"Added {len(documents)} documents to knowledge base")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Simple keyword-based search"""
        query_lower = query.lower()
        results = []
        
        for doc in self.knowledge_base:
            # Simple keyword matching
            text_lower = doc['text'].lower()
            score = 0
            
            # Count keyword matches
            keywords = query_lower.split()
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    'text': doc['text'],
                    'metadata': doc['metadata'],
                    'relevance': score / max(len(keywords), 1)
                })
        
        # Sort by relevance and return top n_results
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:n_results]
    
    def load_initial_knowledge(self):
        """Load initial Data Science knowledge base"""
        knowledge_docs = [
            {
                "text": "Data Science Lifecycle: 1. Problem Definition 2. Data Collection 3. Data Cleaning 4. EDA 5. Feature Engineering 6. Model Selection 7. Model Training 8. Model Evaluation 9. Deployment 10. Monitoring",
                "metadata": {"topic": "Data Science Fundamentals"}
            },
            {
                "text": "Key Machine Learning Algorithms: Supervised Learning: Linear Regression, Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, SVM, Neural Networks. Unsupervised Learning: K-Means Clustering, Hierarchical Clustering, PCA, t-SNE. Evaluation Metrics: MSE, RMSE, MAE, R-squared, Accuracy, Precision, Recall, F1, AUC-ROC",
                "metadata": {"topic": "Machine Learning"}
            },
            {
                "text": "Generative AI Concepts: Core Components: LLMs (Large Language Models): GPT, LLaMA, Claude. Transformers: Self-attention mechanism, positional encoding. Prompt Engineering: Zero-shot, few-shot, chain-of-thought. RAG (Retrieval Augmented Generation): Combining retrieval with generation. Fine-tuning: Adapting pre-trained models to specific tasks.",
                "metadata": {"topic": "Generative AI"}
            },
            {
                "text": "Agentic AI Fundamentals: AI Agents combine LLMs with tools and decision-making. Key Agent Architectures: ReAct (Reason + Act), Plan-and-Execute, Multi-Agent Systems. Components: Planning Module, Memory (short-term and long-term), Tool Use, Reflection. Popular Frameworks: LangChain, AutoGPT, BabyAGI.",
                "metadata": {"topic": "Agentic AI"}
            },
            {
                "text": "Python for Data Science: Essential Libraries: NumPy for numerical computing, Pandas for data manipulation, Matplotlib/Seaborn for visualization, Scikit-learn for machine learning. Key operations: data cleaning, transformation, analysis, and modeling.",
                "metadata": {"topic": "Python Data Science"}
            }
        ]
        
        self.add_documents(knowledge_docs)
        return len(knowledge_docs)