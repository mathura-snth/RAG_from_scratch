"""
eval.py — RAGAS Evaluation for DentaRAG
"""

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from embedding import EmbeddingManager
from vector_store import VectorStore
from retriever import RAGRetriever

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ---------------------------------------------------------------------------
# Dataset de test adapté à vos vrais documents (Biopsie, Composites, Rats)
# Limité à 3 questions pour respecter les quotas gratuits de Groq
# ---------------------------------------------------------------------------
FRENCH_TEST_SET = [
    {
        "question": "Quelle taille de biopsie de la muqueuse orale est recommandée pour optimiser le diagnostic histopathologique ?",
        "ground_truth": (
            "Une longueur de 10mm de l'échantillon semble être adéquate pour optimiser "
            "la pose du diagnostic histopathologique."
        ),
    },
    {
        "question": "Quels sont les mécanismes principaux qui contribuent à l'usure des composites résineux dentaires ?",
        "ground_truth": (
            "Les mécanismes qui contribuent de manière significative au processus d'usure "
            "des composites sont l'usure abrasive (mécanisme prédominant) et l'usure par fatigue."
        ),
    },
    {
        "question": "Comment évolue l'expression de l'ARNm dans le muscle masséter des rats entre la jeunesse et l'âge adulte ?",
        "ground_truth": (
            "Il y a une augmentation de l'expression de l'ARNm des fibres Myh4 (MyHC-IIb) "
            "et Myh1 (MyHC-IIx) dans le muscle masséter chez les rats adultes par rapport aux jeunes rats."
        ),
    }
]

def build_rag_answer(question: str, retriever: RAGRetriever, llm) -> dict:
    retrieved = retriever.retrieve(question, top_k=3)
    contexts = [doc["content"] for doc in retrieved]
    context_text = "\n\n".join(contexts)

    if not context_text:
        answer = "Aucun contexte pertinent trouvé."
    else:
        prompt = (
            "Utilisez le contexte suivant pour répondre à la question.\n\n"
            f"Contexte:\n{context_text}\n\n"
            f"Question: {question}\n\nRéponse:"
        )
        response = llm.invoke(prompt)
        answer = response.content

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts if contexts else [""],
    }

def run_evaluation(output_path=None):
    load_dotenv()

    print("Chargement des modèles...")
    embedding_manager = EmbeddingManager(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = VectorStore(collection_name="pdf_documents", persist_directory="data/vector_store")
    retriever = RAGRetriever(vector_store, embedding_manager)

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="qwen/qwen3.8-27b",
        temperature=0.0,
        max_tokens=500, # Limite abaissée pour éviter le RateLimit de Groq
    )

    print(f"\nExécution du RAG sur {len(FRENCH_TEST_SET)} questions de test...")
    rows = []
    for i, sample in enumerate(FRENCH_TEST_SET):
        print(f"  [{i+1}/{len(FRENCH_TEST_SET)}] {sample['question'][:60]}...")
        row = build_rag_answer(sample["question"], retriever, llm)
        row["ground_truth"] = sample["ground_truth"]
        rows.append(row)

    dataset = Dataset.from_list(rows)

    print("\nCalcul des métriques RAGAS (évaluation via LLM)...")
    eval_embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=eval_embeddings
    )

    scores = results.to_pandas()[
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ].mean().to_dict()

    print("\n" + "=" * 55)
    print("  DentaRAG — RAGAS Evaluation Results")
    print("=" * 55)
    print(f"  Faithfulness      : {scores['faithfulness']:.3f}")
    print(f"  Answer Relevancy  : {scores['answer_relevancy']:.3f}")
    print(f"  Context Precision : {scores['context_precision']:.3f}")
    print(f"  Context Recall    : {scores['context_recall']:.3f}")
    print("=" * 55)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        print(f"\nRésultats sauvegardés dans : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_evaluation(output_path=args.output)