import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from rag_app.retriever import get_mmr_retriever

load_dotenv()
STRICT_RAG_PROMPT = PromptTemplate.from_template(
    """You are a knowledge base assistant. Your job is to answer questions STRICTLY based on the provided context documents.

check the question , perform the actions accordingly. try to follow the rules mentioned below
rules:
1. Answer ONLY using information found in the context below
2. Do NOT use any prior knowledge, training data, or external information.
3. Be concise and factual. Quote or paraphrase from the context directly.
4. If the context partially answers the question, provide what you can and note the limitation.
5. If the question is related to the document summary or the core idea of the document, go through all provided context chunks and summarize them instead of telling it is not present.
6. If the answer is not present in the context, reply with "not present in the knowledge base".

Context:
{context}

Question: {question}

Answer format - do not add the thinking part , just display the answer from the context.

"""
)
def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_filename", doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "")
        page_info = f" (page {page + 1})" if page != "" else ""
        parts.append(f"[Source {i}: {source}{page_info}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in your .env file or environment."
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.3, 
    )


def ask(question: str) -> Dict[str, Any]:
    retriever = get_mmr_retriever()
    llm = _get_llm()

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": "I don't have information about this in the knowledge base.",
            "sources": [],
            "context": "",
        }

    context = _format_docs(retrieved_docs)


    chain = STRICT_RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = []
    seen = set()
    for doc in retrieved_docs:
        filename = doc.metadata.get("source_filename", doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "")
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": filename,
                "page": page + 1 if page != "" else None,
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })

    return {
        "answer": answer,
        "sources": sources,
        "context": context,
    }
