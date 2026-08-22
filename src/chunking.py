from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import uuid
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller chunks and assigns a unique chunk_id.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add unique chunk_id to each chunk
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        
    return chunks
