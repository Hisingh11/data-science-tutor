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