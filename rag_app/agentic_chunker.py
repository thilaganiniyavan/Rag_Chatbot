import os
import json
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import dotenv

class ChunkedProposal(BaseModel):
    chunks: List[str] = Field(description="A list of semantically coherent text chunks.")

class AgenticChunker:
    """
    AgenticChunker uses an LLM to split a document into semantically coherent chunks.
    It identifies natural breakpoints based on content structure and meaning.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.1):
        dotenv.load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature
        )
        
        self.parser = JsonOutputParser(pydantic_object=ChunkedProposal)
        
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert document processor. Your task is to split the following text into semantically coherent chunks.
            
            Instructions:
            1. Analyze the text and identify natural breakpoints (e.g., changes in topic, section transitions, end of paragraphs).
            2. Each chunk should be self-contained and maintain its own context.
            3. Do NOT change the original wording of the text.
            4. The chunks should be roughly 400-800 characters long, but prioritize semantic coherence over exact length.
            5. Return the result as a JSON object with a key "chunks" containing the list of strings.

            Text to split:
            {text}

            {format_instructions}
            """
        )

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using the LLM."""
        if not text.strip():
            return []
            
        # For very long texts, we might need to do initial broad splitting 
        # but for now we'll process the input text as a single unit or in large blocks.
        # This implementation assumes the input text per document is manageable.
        
        chain = self.prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({
                "text": text,
                "format_instructions": self.parser.get_format_instructions()
            })
            return result.get("chunks", [])
        except Exception as e:
            # Fallback or error handling
            print(f"Error during agentic chunking: {e}")
            # If LLM fails, return the text as a single chunk to avoid data loss
            return [text]

    def split_documents(self, documents: List) -> List:
        """Compatibility method for LangChain's split_documents."""
        from langchain_core.documents import Document
        
        final_chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            for i, chunk_text in enumerate(text_chunks):
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i
                final_chunks.append(Document(page_content=chunk_text, metadata=metadata))
        
        return final_chunks
