import os
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from rag_app.agentic_chunker import AgenticChunker


def _get_loader(file_path: str, file_extension: str):
    """Return the appropriate LangChain loader based on file extension."""
    ext = file_extension.lower().lstrip(".")

    if ext == "pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path)

    elif ext in ("docx", "doc"):
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path)

    elif ext == "txt":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8")

    elif ext == "csv":
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(file_path, encoding="utf-8")

    elif ext == "json":
        from langchain_community.document_loaders import JSONLoader
        return JSONLoader(file_path, jq_schema=".", text_content=False)

    elif ext in ("html", "htm"):
        from langchain_community.document_loaders import BSHTMLLoader
        return BSHTMLLoader(file_path, open_encoding="utf-8")

    elif ext in ("md", "markdown"):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8")

    elif ext in ("xlsx", "xls"):
        from langchain_community.document_loaders import UnstructuredExcelLoader
        return UnstructuredExcelLoader(file_path)

    elif ext in ("pptx", "ppt"):
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        return UnstructuredPowerPointLoader(file_path)

    elif ext == "xml":
        from langchain_community.document_loaders import UnstructuredXMLLoader
        return UnstructuredXMLLoader(file_path)

    elif ext == "sql":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8")

    elif ext == "db":
        from langchain_community.document_loaders import SQLDatabaseLoader
        from langchain_community.utilities.sql_database import SQLDatabase
        # Create a temporary SQLite engine for the uploaded file
        db = SQLDatabase.from_uri(f"sqlite:///{file_path}")
        return SQLDatabaseLoader.from_query("SELECT * FROM sqlite_master WHERE type='table'", db)

    else:
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8")


def load_and_chunk_file(file_path: str, original_filename: str) -> List[Document]:
    ext = Path(original_filename).suffix
    loader = _get_loader(file_path, ext)

    try:
        raw_docs = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to load '{original_filename}': {e}")

    for doc in raw_docs:
        doc.metadata["source_filename"] = original_filename

    chunker = AgenticChunker()
    chunks = chunker.split_documents(raw_docs)
    return chunks


def load_and_chunk_bytes(file_bytes: bytes, original_filename: str) -> List[Document]:
    ext = Path(original_filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        chunks = load_and_chunk_file(tmp_path, original_filename)
    finally:
        os.unlink(tmp_path)

    return chunks
