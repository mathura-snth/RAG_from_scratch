import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_groq import ChatGroq
from config import DATA_DIR

# Importer les modules du projet
from embedding import EmbeddingManager
from vector_store import VectorStore
from retriever import RAGRetriever

# --- Initialisation ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initialisation du RAG en cours...")
embedding_manager = EmbeddingManager()
import os
vector_store = VectorStore(persist_directory=os.path.join(DATA_DIR, "vector_store"))
rag_retriever = RAGRetriever(vector_store, embedding_manager)

# Chargement de la clé API et initialisation de Groq
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="qwen/qwen3.8-27b",
    temperature=0.1,
    max_tokens=1024
)
print("RAG prêt !")

# --- Routes ---
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    retrieved_docs = rag_retriever.retrieve(request.question, top_k=3)
    
    sources = []
    contexts = []
    for doc in retrieved_docs:
        sources.append({
            "source": doc["metadata"].get("source", "Inconnu"),
            "content": doc["content"][:250] + "..."
        })
        contexts.append(doc["content"])
        
    context_text = "\n\n".join(contexts)
    
    if not context_text:
        final_answer = "Je n'ai trouvé aucune information pertinente pour répondre à votre question."
    else:
        prompt = f"""Utilise le contexte suivant pour répondre de manière concise à la question.

Context:
{context_text}

Question: {request.question}
Answer:"""
        response = llm.invoke(prompt)
        final_answer = response.content
    
    return {
        "answer": final_answer,
        "sources": sources
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)