

import os
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_app.embeddings import get_embeddings

PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "rag_knowledge_base"

_vectorstore_instance = None


def get_vectorstore() -> Chroma:
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=PERSIST_DIR,
        )
    return _vectorstore_instance


def add_documents(documents: List[Document]) -> int:
    vs = get_vectorstore()
    vs.add_documents(documents)
    return len(documents)


def get_collection_info() -> Dict[str, Any]:

    vs = get_vectorstore()
    count = vs._collection.count()
    return {
        "count": count,
        "persist_dir": PERSIST_DIR,
    }


def clear_vectorstore() -> None:
    global _vectorstore_instance
    vs = get_vectorstore()
    vs.delete_collection()
    _vectorstore_instance = None  
