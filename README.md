# RAG Application

A Retrieval-Augmented Generation (RAG) system built with Streamlit, LangChain, and Groq. This application allows users to upload documents and query their contents using advanced retrieval techniques and large language models.

## Features

- Multi-format Document Support.d TXT files.
- Advanced Retrieval: Uses Maximal Marginal Relevance (MMR) to ensure diverse and relevant results.
- Strict Groundedness: Answers are strictly derived from the provided context to minimize hallucinations.
- Modern Web Interface: Built with Streamlit for a fast and interactive user experience.
- High-Performance LLM: Powered by llama-3.3-70b-versatile via the Groq API.
- Local Embeddings: Uses sentence-transformers/all-MiniLM-L6-v2 for efficient local vector generation.
- Persistent Vector Store: ChromaDB is used to store and manage document embeddings.

## Project Structure

- `app.py`: Main Streamlit application and UI logic.
- `rag_app/`:
  - `document_processor.py`: Handles document loading and chunking.
  - `embeddings.py`: Provides a singleton wrapper for local embeddings.
  - `vector_store.py`: Manages the ChromaDB instance and document indexing.
  - `retriever.py`: Sets up the MMR-based retriever.
  - `rag_chain.py`: Configures the RAG prompt and execution chain.

## Prerequisites

- Python 3.10 or higher.
- A Groq API Key.

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

Start the Streamlit application:
```bash
streamlit run app.py
```

Access the application in your web browser at `http://127.0.0.1:8501`.


