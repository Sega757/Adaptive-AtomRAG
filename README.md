# Adaptive-AtomRAG ⚡️

**Sub-millisecond query routing & cost-aware execution for GraphRAG architectures.**

Adaptive-AtomRAG is a highly optimized conditional routing gateway designed to resolve the latency and cost bottlenecks of heavy Atom-Entity Graph (AEG) RAG systems. By integrating a lightweight `TF-IDF + SVM` classifier, the system dynamically routes queries, reducing average latency by 47% and token costs by 48% on simple tasks.

## 🚀 Key Features

* **Sub-millisecond Routing:** Classification takes `< 1 ms`, preventing unnecessary LLM planning overhead.
* **Cost-Aware Execution (CA-RAG):** Maximizes utility using a customizable scalar function balancing quality, latency, and token cost.
* **Anti-Noise Context:** Prevents context pollution on simple factual queries by bypassing deep graph traversal.
* **Production Ready:** Built on FastAPI with asynchronous endpoints, ready for Prometheus metrics and Kubernetes deployment.

## 📊 Benchmarks & System Impact

| Metric | Naive RAG / Direct | AtomRAG (Unconditional) | Adaptive-AtomRAG (Routed) | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **P95 Latency** | ~4.0s | ~22.3s | **Dynamic (52.9% of base)** | **-47.1%** |
| **Token Cost** | Fixed | Maximum | **Optimized** | **-48.5%** |
| **Routing F1 (Macro)**| N/A | N/A | **0.928** | TF-IDF + SVM |

## 🧠 Mathematical Foundation

The router maximizes the query utility $U_b$ for a given bundle $b$:

$$U_b = w_Q \hat{Q}_b(q) - w_L \hat{L}_b^{\text{norm}} - w_C \hat{C}_b^{\text{norm}}$$

To prevent *router collapse* during token-level routing, sparse regularization penalties $\lambda$ are applied to the loss gradient:

$$\frac{\partial d_t}{\partial L} = \left(\frac{\partial o_t}{\partial L_{\text{NTP}}}\right)^T a_t + 2\lambda d_t$$

## 🛠 Quick Start

```bash
# Clone the repository
git clone [https://github.com/YourUser/Adaptive-AtomRAG.git](https://github.com/YourUser/Adaptive-AtomRAG.git)
cd Adaptive-AtomRAG

# Spin up the router
docker-compose up -d --build


Adaptive-AtomRAG/
├── README.md                  # Витрина проекта
├── requirements.txt           # Зависимости
├── docker-compose.yml         # Быстрый запуск
├── docs/
│   └── whitepaper.md          # Твоя статья с формулами и бенчмарками
└── router/
    ├── __init__.py
    ├── classifier.py          # Логика TF-IDF + SVM маршрутизатора
    └── main.py                # FastAPI шлюз