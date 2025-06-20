# PDF Chatbot (Local Embeddings & Local LLM)

## Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about a PDF using local embeddings and a local open-source LLM (TinyLlama). No paid APIs or cards required.

## Features
- Document chunking and semantic embedding (sentence-transformers)
- Vector database (Chroma)
- Local LLM answer generation (TinyLlama)
- Real-time streaming responses in Streamlit
- Source chunk display
- Model/chunk info in sidebar
- Clear chat/reset functionality

## How to Run

1. Install requirements:
    ```
    pip install -r requirements.txt
    pip install transformers
    ```

2. Chunk your PDF:
    ```
    python src/chunker.py
    ```

3. Embed chunks:
    ```
    python src/embedder.py
    ```

4. Start the app:
    ```
    streamlit run app.py
    ```

## Example Queries

- What is the main topic of the document?
- What sections survive after the termination of the User Agreement?
- What is the process for resolving legal disputes according to the document?

## Screenshots

(Add your screenshots here)

---

## Model & Embedding Choices

- Embeddings: all-MiniLM-L6-v2 (sentence-transformers)
- LLM: TinyLlama-1.1B-Chat-v1.0 (Hugging Face, runs locally)

---

## Notes

- All code is free and runs locally.
- No API keys or payment required.