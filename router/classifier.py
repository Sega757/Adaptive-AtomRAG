import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from typing import Tuple

class ComplexityRouter:
    """
    Легковесный TF-IDF + SVM классификатор сложности запросов.
    Обеспечивает задержку < 1 ms для предотвращения overhead'а.
    """
    def __init__(self):
        # Имитация обученного векторайзера и классификатора (в проде грузится из .onnx или .pkl)
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.classifier = LinearSVC(C=1.0, class_weight='balanced', dual=False)
        
        # Калибровочные данные для холодного старта
        dummy_corpus = [
            "What is the capital of France?", # 0: Simple
            "Summarize the Q3 financial report.", # 1: Medium
            "How does the geopolitical situation in region X affect the supply chain of company Y considering Z regulations?" # 2: Complex
        ]
        dummy_labels = [0, 1, 2]
        
        # Предварительное обучение
        X = self.vectorizer.fit_transform(dummy_corpus)
        self.classifier.fit(X, dummy_labels)
        
    def route_query(self, query: str) -> Tuple[int, float]:
        """
        Определяет оптимальный путь выполнения RAG.
        0: No Retrieval (Direct LLM)
        1: Single-step Retrieval (Naive RAG)
        2: Multi-step Graph Retrieval (AtomRAG)
        """
        start_time = time.perf_counter()
        
        vec = self.vectorizer.transform([query])
        prediction = self.classifier.predict(vec)[0]
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return int(prediction), execution_time_ms

# Singleton instance
router_engine = ComplexityRouter()