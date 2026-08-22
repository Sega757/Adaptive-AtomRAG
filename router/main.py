from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from router.classifier import router_engine

app = FastAPI(
    title="Adaptive-AtomRAG Router API",
    description="Sub-millisecond query routing gateway for Cost-Aware RAG.",
    version="1.0.0"
)

# Автоматический сбор метрик для Prometheus
Instrumentator().instrument(app).expose(app)

class QueryRequest(BaseModel):
    query: str
    max_latency_budget_ms: int = 2000

class RoutingResponse(BaseModel):
    route_class: int
    route_name: str
    routing_latency_ms: float
    recommended_engine: str

ROUTE_MAP = {
    0: ("Simple Fact", "Direct LLM Cache"),
    1: ("Single-Step", "Vector Search (HNSW)"),
    2: ("Multi-Step Complex", "Atom-Entity Graph (PageRank)")
}

@app.post("/api/v1/route", response_model=RoutingResponse)
async def evaluate_query(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    route_id, calc_time = router_engine.route_query(payload.query)
    route_name, engine = ROUTE_MAP.get(route_id, ROUTE_MAP[1])
    
    return RoutingResponse(
        route_class=route_id,
        route_name=route_name,
        routing_latency_ms=round(calc_time, 4),
        recommended_engine=engine
    )

@app.get("/health")
async def health_check():
    return {"status": "operational", "router_type": "TF-IDF + LinearSVC"}