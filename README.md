# PDF_RAG

A Retrieval-Augmented Generation (RAG) pipeline for PDF documents. It extracts text from PDFs, chunks them, creates vector embeddings (ChromaDB) and BM25 search indices, and uses a Groq-powered LLM to generate precise answers to user queries based on the document context.

## Features
- **Extraction:** PyMuPDF for parsing text from PDFs.
- **Chunking:** Langchain text splitters for optimal document segmentation.
- **Retrieval:** Hybrid search using ChromaDB (vector embeddings) and BM25 (keyword search), with reranking.
- **Generation:** Groq API for fast LLM inference.
- **Environment Management:** Uses python-dotenv for secure API key management.

## Installation
1. Clone the repository
2. Install dependencies: 
```bash
pip install -r requirements.txt
```
3. Create a `.env` file and add your `GROQ_API_KEY`.
4. Place your PDF files in the `data/` directory.

## Usage
Run the main script to index your documents and start the interactive Q&A session:
```bash
python main.py
```
## Code Readability
The codebase is thoroughly commented to make the implementation and RAG pipeline easier to understand and follow.
