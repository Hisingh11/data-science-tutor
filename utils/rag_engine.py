import os
import json
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st

class RAGEngine:
    def __init__(self, persist_directory="./data/knowledge_base"):
        """Initialize RAG engine with ChromaDB and embedding model"""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding model (free, local)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection_name = "ds_knowledge"
        collections = [c.name for c in self.chroma_client.list_collections()]
        
        if self.collection_name not in collections:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Data Science knowledge base"}
            )
        else:
            self.collection = self.chroma_client.get_collection(self.collection_name)
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to vector store"""
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}_{hash(doc['text'])}"
            ids.append(doc_id)
            texts.append(doc['text'])
            metadatas.append(doc.get('metadata', {}))
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        st.success(f"Added {len(documents)} documents to knowledge base")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                doc_data = {
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                    'relevance': 1.0
                }
                
                # Add relevance score if distances are available
                if results['distances'] and results['distances'][0]:
                    doc_data['relevance'] = 1 - results['distances'][0][i]
                
                documents.append(doc_data)
        
        return documents
    
    def load_initial_knowledge(self):
        """Load initial Data Science knowledge base"""
        knowledge_docs = [
            {
                "text": "Data Science Lifecycle: 1. Problem Definition: Understanding business objectives and requirements. 2. Data Collection: Gathering data from various sources. 3. Data Cleaning: Handling missing values, outliers, inconsistencies. 4. EDA: Exploratory Data Analysis for pattern discovery. 5. Feature Engineering: Creating relevant features for modeling. 6. Model Selection: Choosing appropriate algorithms. 7. Model Training: Fitting models to training data. 8. Model Evaluation: Testing on validation/holdout sets. 9. Deployment: Putting models into production. 10. Monitoring: Tracking model performance over time.",
                "metadata": {"topic": "Data Science Fundamentals"}
            },
            {
                "text": "Key Machine Learning Algorithms: Supervised Learning: Linear Regression (for continuous target variables), Logistic Regression (binary classification), Decision Trees (interpretable, handles non-linear), Random Forest (ensemble, reduces overfitting), Gradient Boosting (XGBoost, LightGBM - high performance), SVM (effective for high-dimensional), Neural Networks (deep learning). Unsupervised Learning: K-Means Clustering, Hierarchical Clustering, PCA (dimensionality reduction), t-SNE (visualization). Evaluation Metrics: Regression: MSE, RMSE, MAE, R-squared. Classification: Accuracy, Precision, Recall, F1, AUC-ROC.",
                "metadata": {"topic": "Machine Learning"}
            },
            {
                "text": "Generative AI Concepts: Core Components: LLMs (Large Language Models): GPT, LLaMA, Claude. Transformers: Self-attention mechanism, positional encoding. Prompt Engineering: Zero-shot, few-shot, chain-of-thought. RAG (Retrieval Augmented Generation): Combining retrieval with generation. Fine-tuning: Adapting pre-trained models to specific tasks. Popular Gen AI Applications: Text Generation: ChatGPT, Claude, Gemini. Image Generation: DALL-E, Stable Diffusion, Midjourney. Code Generation: GitHub Copilot, CodeLlama. Key Papers: Attention is All You Need (Transformer architecture), BERT, Language Models are Few-Shot Learners (GPT-3).",
                "metadata": {"topic": "Generative AI"}
            },
            {
                "text": "Agentic AI Fundamentals: What are AI Agents? Autonomous systems that perceive their environment and take actions to achieve goals. Combine LLMs with tools and decision-making capabilities. Key Agent Architectures: 1. ReAct (Reason + Act): Iterative reasoning and action taking. 2. Plan-and-Execute: Planning before execution. 3. Multi-Agent Systems: Multiple specialized agents collaborating. Essential Agent Components: Planning Module: Breaks down complex tasks. Memory: Short-term (conversation) and long-term (vector stores). Tool Use: API calls, code execution, web search. Reflection: Self-evaluation and improvement. Popular Agent Frameworks: LangChain, AutoGPT, BabyAGI.",
                "metadata": {"topic": "Agentic AI"}
            },
            {
                "text": "Python for Data Science - Essential Libraries: NumPy: Numerical computing with ndarray, broadcasting, linear algebra. Pandas: Data manipulation with DataFrame, Series, data cleaning (dropna, fillna, replace), group operations (groupby, aggregate), merging (merge, concat). Matplotlib/Seaborn: Visualization with line plots, scatter plots, bar charts, heatmaps, pairplots. Scikit-learn: Machine learning with train-test split, cross-validation, preprocessing (StandardScaler, MinMaxScaler), models (LinearRegression, RandomForest, XGBoost), metrics (classification_report, confusion_matrix).",
                "metadata": {"topic": "Python Data Science"}
            }
        ]
        
        self.add_documents(knowledge_docs)
        return len(knowledge_docs)