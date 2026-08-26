from typing import List, Dict, Any

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = -1.0) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Process results
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_id, document, metadata, distance) in enumerate(
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    zip(results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0])
                ):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        # we add it inside the context documents which is created in retrieved_docs 
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []