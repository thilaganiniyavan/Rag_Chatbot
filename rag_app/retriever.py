
from langchain_core.vectorstores import VectorStoreRetriever

from rag_app.vector_store import get_vectorstore

MMR_K = 5
MMR_FETCH_K = 20
MMR_LAMBDA = 0.5


def get_mmr_retriever() -> VectorStoreRetriever:

    vs = get_vectorstore()
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": MMR_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )
    return retriever
