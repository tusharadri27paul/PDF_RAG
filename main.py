import os
from pathlib import Path
from src.config import DATA_DIR, DB_DIR
from src.extraction import extract_documents_from_pdf
from src.chunking import chunk_documents
from src.indexing import RAGIndexer
from src.retrieval import RAGRetriever
from src.generation import RAGGenerator

def ingest_pdfs():
    print("--- Starting Document Ingestion ---")
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}. Please add some PDFs in that folder and run again.")
        return False
        
    all_chunks = []
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        
        # 1. Extraction
        documents = extract_documents_from_pdf(str(pdf_path))
        print(f"Extracted {len(documents)} sections from {pdf_path.name}.")
        
        # 2. Chunking
        chunks = chunk_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        
        all_chunks.extend(chunks)
        
    # 3. Indexing (ChromaDB + BM25)
    indexer = RAGIndexer()
    indexer.index_documents(all_chunks)
    print("--- Ingestion Complete ---")
    return True

def query_pipeline(query: str):
    print(f"\n--- Processing Query: '{query}' ---")
    
    # 4. Retrieval & Reranking
    try:
        retriever = RAGRetriever()
        print("Retrieving and reranking documents...")
        top_chunks = retriever.retrieve_and_rerank(query, final_k=5)
        
        if not top_chunks:
            print("No relevant context found in the PDFs.")
            return
            
        print(f"Found {len(top_chunks)} highly relevant chunks.")
        
        # 5. Generation
        print("Generating answer using Groq...")
        generator = RAGGenerator()
        answer = generator.generate_answer(query, top_chunks)
        
        print("\n=== AI ANSWER ===")
        print(answer)
        print("=================\n")
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    bm25_path = DB_DIR / "bm25_index.pkl"
    
    if not bm25_path.exists():
        print("No existing index found. Training model from PDFs...")
        success = ingest_pdfs()
        if not success:
            exit(1)
    else:
        print("Existing index found. Loading directly (Skipping training step).")
        
    print("\nModel is ready! You can now ask questions about your PDFs.")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            query = input("\nYour Question: ").strip()
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            if not query:
                continue
                
            query_pipeline(query)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
