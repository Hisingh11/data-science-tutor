import time

from utils.model_manager import get_model_manager
from utils.rag_engine import RAGEngine
from utils.self_rag_graph import SelfRAGEngine

rag = RAGEngine()
rag.load_initial_knowledge()
model = get_model_manager()
engine = SelfRAGEngine(rag, model)
start = time.time()
result = list(engine.iter_answer('What is overfitting?', 'What is overfitting?', []))
elapsed = time.time() - start
print(f'Time: {elapsed:.2f}s')
print(f'Answer length: {len(str(result))}')
print(f'Calls: {engine.last_log}')
