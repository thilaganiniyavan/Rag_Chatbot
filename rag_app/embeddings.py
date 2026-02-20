"""
Embeddings: Singleton wrapper around HuggingFace sentence-transformers.
Uses all-MiniLM-L6-v2 (384 dimensions) — no API key required.
"""

from langchain_huggingface import HuggingFaceEmbeddings

_embeddings_instance = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a singleton HuggingFaceEmbeddings instance.
    The model is loaded once and reused across all calls.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance
