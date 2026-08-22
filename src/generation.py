from groq import Groq
from .config import GROQ_API_KEY
from typing import List, Dict, Any

class RAGGenerator:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "openai/gpt-oss-120b" # Groq free tier fast model
        
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        # Build context string with citations
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            text = chunk['text']
            page = chunk['metadata'].get('page_number', 'Unknown')
            source = chunk['metadata'].get('source', f'Source {i+1}')
            section = chunk['metadata'].get('section_heading', 'Unknown')
            
            context_parts.append(f"[{source} | Page: {page} | Section: {section}]\n{text}")
            
        context_str = "\n\n".join(context_parts)
        
        system_prompt = (
            "You are an expert AI assistant answering questions based strictly on the provided context.\n"
            "If the answer is not contained in the context, state that clearly.\n"
            "When providing facts or details, always cite the source using the provided file name and Page numbers exactly in this format: 【file_name.pdf | Page 5】.\n"
            "Format your answer beautifully using Markdown.\n"
        )
        
        user_prompt = f"Context Information:\n{context_str}\n\nUser Question: {query}"
        
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            model=self.model,
            temperature=0.0, # Deterministic answers
        )
        
        return response.choices[0].message.content
