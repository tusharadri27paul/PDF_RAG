import fitz
from langchain_core.documents import Document
from typing import List
from pathlib import Path

def extract_documents_from_pdf(pdf_path: str) -> List[Document]:
    """
    Extracts text and metadata (page number, heading) from a PDF.
    Groups text by section/heading to create Langchain Documents.
    """
    doc = fitz.open(pdf_path)
    documents = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        
        current_heading = "General" # Default heading
        section_text = []
        
        for b in blocks:
            if b['type'] == 0:  # Text block
                for l in b["lines"]:
                    for s in l["spans"]:
                        text = s["text"].strip()
                        if not text:
                            continue
                        
                        font_size = s["size"]
                        
                        # Simple heuristic for heading: bold or large font
                        is_bold = "bold" in s["font"].lower()
                        if font_size > 13 or (font_size > 11 and is_bold):
                            # If we have accumulated text for a previous heading, save it
                            if section_text:
                                documents.append(
                                    Document(
                                        page_content=" ".join(section_text),
                                        metadata={
                                            "source": Path(pdf_path).name,
                                            "page_number": page_num + 1,
                                            "section_heading": current_heading
                                        }
                                    )
                                )
                                section_text = []
                            current_heading = text
                        else:
                            section_text.append(text)
                            
        # Append remaining text at the end of the page
        if section_text:
            documents.append(
                Document(
                    page_content=" ".join(section_text),
                    metadata={
                        "source": Path(pdf_path).name,
                        "page_number": page_num + 1,
                        "section_heading": current_heading
                    }
                )
            )
            
    return documents
