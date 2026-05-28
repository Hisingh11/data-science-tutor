import os
import json
from typing import List, Dict
import chromadb
import streamlit as st
import numpy as np

# Simple embedding function without torch
class SimpleEmbeddingFunction:
    def __init__(self):
        self.dimension = 384
    
    def __call__(self, texts):
        # Simple hash-based embeddings (for demo purposes)
        embeddings = []
        for text in texts:
            hash_val = hash(text) % 10000
            np.random.seed(hash_val)
            emb = np.random.randn(self.dimension).astype(np.float32)
            embeddings.append(emb.tolist())
        return embeddings

class RAGEngine:
    def __init__(self, persist_directory="./data/knowledge_base"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Use simple embedding function instead of sentence-transformers
        self.embedding_function = SimpleEmbeddingFunction()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection_name = "ds_knowledge"
        collections = [c.name for c in self.chroma_client.list_collections()]
        
        if self.collection_name not in collections:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Data Science knowledge base"}
            )
        else:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to vector store - silently"""
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}_{abs(hash(doc['text']))}"
            ids.append(doc_id)
            texts.append(doc['text'])
            metadatas.append(doc.get('metadata', {}))
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        # Removed st.success to keep UI clean
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'relevance': 1.0
                })
        
        return documents
    
    def load_initial_knowledge(self):
        knowledge_docs = [
            {"text": "Data Science Lifecycle: Problem Definition, Data Collection, Data Cleaning, EDA, Feature Engineering, Model Selection, Training, Evaluation, Deployment, Monitoring", "metadata": {"topic": "Data Science"}},
            {"text": "Machine Learning: Supervised (Regression, Classification), Unsupervised (Clustering, PCA), Evaluation Metrics (MSE, Accuracy, F1, AUC-ROC)", "metadata": {"topic": "ML"}},
            {"text": "Generative AI: LLMs, Transformers, Attention Mechanism, Prompt Engineering, RAG, Fine-tuning", "metadata": {"topic": "GenAI"}},
            {"text": "Agentic AI: AI Agents, ReAct Pattern, Planning, Memory, Tool Use, Multi-agent Systems", "metadata": {"topic": "Agentic AI"}},
            {"text": "Python for Data Science: NumPy, Pandas, Matplotlib, Scikit-learn for data manipulation and ML", "metadata": {"topic": "Python"}}
        ]
        self.add_documents(knowledge_docs)
        return len(knowledge_docs)