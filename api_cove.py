"""
api.py — DentaRAG FastAPI server (CoVe pipeline)
=================================================
All questions go through the Factored Chain-of-Verification pipeline.
The response includes the intermediate CoVe steps for transparency.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_groq import ChatGroq

from embedding import EmbeddingManager
from vector_store import VectorStore
from retriever import RAGRetriever
from cove import CoVeEngine

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

load_dotenv()

app = FastAPI(
    title="DentaRAG API",
    description="RAG + Chain-of-Verification pipeline over dental research documents",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initialising DentaRAG (CoVe pipeline)...")

embedding_manager = EmbeddingManager()
vector_store = VectorStore(persist_directory="data/vector_store")
rag_retriever = RAGRetriever(vector_store, embedding_manager)

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="qwen/qwen3-32b",
    temperature=0.1,
    max_tokens=1024,
)

cove_engine = CoVeEngine(llm=llm, n_verification_questions=3)

_startup_time = datetime.now()
print("DentaRAG ready !")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Returns API status, document count, and uptime."""
    return {
        "status": "ok",
        "pipeline": "Factored CoVe",
        "documents_indexed": vector_store.collection.count(),
        "embedding_model": embedding_manager.model_name,
        "llm_model": llm.model_name,
        "uptime_seconds": (datetime.now() - _startup_time).seconds,
    }


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    # Final answer (after CoVe revision)
    answer: str
    # Retrieved source chunks
    sources: list[dict]
    # CoVe intermediate steps (useful for demo / explainability)
    cove_steps: dict


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Query the RAG + CoVe pipeline.

    Returns the final verified answer plus the intermediate CoVe steps
    (draft, verification questions and answers) for full transparency.
    """
    # 1. Retrieval
    retrieved_docs = rag_retriever.retrieve(request.question, top_k=5)

    sources = []
    contexts = []
    for doc in retrieved_docs:
        sources.append({
            "source": doc["metadata"].get("source", "Unknown"),
            "score": round(doc["similarity_score"], 4),
            "content": doc["content"][:250] + "...",
        })
        contexts.append(doc["content"])

    # 2. CoVe pipeline (draft → plan → execute → revise)
    if not contexts:
        return ChatResponse(
            answer="No relevant information found in the knowledge base.",
            sources=[],
            cove_steps={},
        )

    cove_result = cove_engine.run(
        question=request.question,
        contexts=contexts,
    )

    return ChatResponse(
        answer=cove_result["final_answer"],
        sources=sources,
        cove_steps={
            "draft": cove_result["draft"],
            "verification_questions": cove_result["verification_questions"],
            "verification_answers": cove_result["verification_answers"],
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
