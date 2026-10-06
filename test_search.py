from utils.rag_engine import RAGEngine
from utils.self_rag_graph import SelfRAGEngine, MAX_RETRIES
from utils.model_manager import get_model_manager

rag = RAGEngine()
rag.load_initial_knowledge()
model = get_model_manager()
# Use the default MAX_RETRIES from the module (now 1)
engine = SelfRAGEngine(rag, model)
import time
start = time.time()
result = list(engine.iter_answer('What is overfitting?', 'What is overfitting?', []))
elapsed = time.time() - start
print(f'Time: {elapsed:.2f}s')
print(f'Answer length: {len(str(result))}')
print(f'Calls: {engine.last_log}')
print(f'Max retries from module: {MAX_RETRIES}')