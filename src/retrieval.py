import pickle
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from .config import DB_DIR, EMBEDDING_MODEL, RERANKER_MODEL
from typing import List, Dict, Any

class RAGRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        
        self.chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.chroma_client.get_collection(name="rag_collection")
        
        self.bm25_path = DB_DIR / "bm25_index.pkl"
        self.chunks_path = DB_DIR / "chunks.pkl"
        
        self.bm25 = None
        self.all_chunks = None
        
        if self.bm25_path.exists() and self.chunks_path.exists():
            with open(self.bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(self.chunks_path, 'rb') as f:
                self.all_chunks = pickle.load(f)
                
    def _reciprocal_rank_fusion(self, vector_results, bm25_results, k=60):
        """Merges lists using RRF."""
        fused_scores = {}
        
        for rank, chunk_id in enumerate(vector_results):
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0
            fused_scores[chunk_id] += 1 / (rank + k)
            
        for rank, chunk_id in enumerate(bm25_results):
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0
            fused_scores[chunk_id] += 1 / (rank + k)
            
        # Sort by fused score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, score in sorted_results]

    def hybrid_search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.all_chunks:
            raise ValueError("Index not found. Please run the indexing pipeline first.")
            
        # 1. Vector Search
        query_embedding = self.embedding_model.encode([query]).tolist()
        vector_res = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        vector_ids = vector_res['ids'][0] if vector_res['ids'] else []
        
        # 2. BM25 Keyword Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get top_k indices
        top_bm25_indices = bm25_scores.argsort()[::-1][:top_k]
        bm25_ids = [self.all_chunks[i].metadata["chunk_id"] for i in top_bm25_indices]
        
        # 3. Merge results using RRF
        fused_ids = self._reciprocal_rank_fusion(vector_ids, bm25_ids)
        
        # Get actual chunks for the fused ids (top 15)
        fused_ids = fused_ids[:top_k]
        
        candidates = []
        for chunk in self.all_chunks:
            if chunk.metadata["chunk_id"] in fused_ids:
                candidates.append({
                    "text": chunk.page_content,
                    "metadata": chunk.metadata
                })
                
        return candidates

    def retrieve_and_rerank(self, query: str, final_k: int = 5) -> List[Dict[str, Any]]:
        # Get top 15 from hybrid search
        candidates = self.hybrid_search(query, top_k=15)
        
        if not candidates:
            return []
            
        # Prepare pairs for cross-encoder
        pairs = [[query, candidate["text"]] for candidate in candidates]
        
        # Predict scores
        scores = self.reranker.predict(pairs)
        
        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
            
        ranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        return ranked_candidates[:final_k]
