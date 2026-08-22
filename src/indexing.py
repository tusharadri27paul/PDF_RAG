import os
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List
from langchain_core.documents import Document
from .config import DB_DIR, EMBEDDING_MODEL

class RAGIndexer:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.bm25_path = DB_DIR / "bm25_index.pkl"
        self.chunks_path = DB_DIR / "chunks.pkl"
        
    def index_documents(self, chunks: List[Document]):
        if not chunks:
            print("No chunks to index.")
            return

        print(f"Indexing {len(chunks)} chunks into ChromaDB...")
        
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings (this might take a moment)...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True).tolist()
        
        # Add to ChromaDB
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("ChromaDB indexing complete.")
        
        # Build BM25 Index
        print("Building BM25 index...")
        tokenized_corpus = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Save BM25 index and chunks for retrieval
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(bm25, f)
            
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
            
        print("BM25 indexing complete.")
