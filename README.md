# Enterprise Retrieval-Augmented Generation (RAG) Assistant

This project is a local RAG assistant that allows users to query a knowledge base of PDF documents using a local LLM.

## Architecture

1.  **Ingestion**: Parses text from PDFs in the `data/` directory.
2.  **Embeddings**: Converts text chunks into vector embeddings using `all-MiniLM-L6-v2`.
3.  **Vector Store**: Stores embeddings in a local ChromaDB.
4.  **Retrieval**: Searches for relevant context based on user queries.
5.  **Generation**: Uses a local LLM (Mistral via Ollama) to generate answers.
6.  **Interface**: Provides a chat interface using Streamlit.

## Setup

1.  **Install Dependencies**:
    This project uses `uv` for dependency management.
    ```bash
    uv sync
    ```

2.  **Install Ollama**:
    Follow the instructions at [ollama.com](https://ollama.com).
    Pull the model:
    ```bash
    ollama pull mistral
    ```

3.  **Prepare Data**:
    Place your PDF documents in the `data/` directory.

4.  **Build Database**:
    ```bash
    uv run build_db.py
    ```

5.  **Run Application**:
    ```bash
    uv run streamlit run app.py
    ```

## Files

*   `ingest.py`: Handles loading and parsing of PDFs.
*   `build_db.py`: Chunks text and builds the vector database.
*   `rag_backend.py`: Defines the RAG chain (Retrieval + Generation).
*   `app.py`: The Streamlit web application.
