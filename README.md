# LangGraph RAG Chatbot (Gemini + ChromaDB + PDF Ingestion)

A fully production-ready, CLI-based Retrieval-Augmented Generation (RAG) chatbot using:

- [LangGraph](https://github.com/langchain-ai/langgraph) for conversation flow
- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Gemini](https://ai.google.dev/) (via `google-generativeai`)
- [ChromaDB](https://www.trychroma.com/) for local, persistent vector storage
- PDF ingestion via PyPDFLoader
- Gemini for both LLM and Embeddings

## Features

- **Understands greetings and responds conversationally**
- **Answers ONLY from ingested PDF content**
- **Refuses to hallucinate/outside-knowledge questions with exact fallback message**

> “Sorry, this question is outside my knowledge base.”

- **Strict conversation flow implemented via LangGraph**
- **PDF ingestion: chunks, page source metadata, Gemini embeddings, persistent ChromaDB**
- **Strict system prompt and retrieval threshold to prevent LLM hallucinations**
- **Clean CLI chat loop ("exit" to quit)**

---

## Setup

1. **Clone repository and install dependencies**

    ```sh
    pip install -r requirements.txt
    ```

2. **Set your Google Gemini API key**

    ```sh
    export GEMINI_API_KEY=your_api_key_here
    ```
    - You can get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

3. **Place PDFs in the `pdfs/` folder**
    - Any PDFs placed here will be ingested and searchable.

4. **Run the chatbot**

    ```sh
    python main.py
    ```

---

## File Structure

```
.
├── chatbot
│   ├── __init__.py
│   ├── llm.py              # Gemini LLM wrapper for strict context answers
│   ├── embed.py            # Gemini embeddings setup
│   ├── vectordb.py         # ChromaDB creation, persistence, search
│   ├── ingest.py           # PDF loading, chunking, ingestion
│   └── flow.py             # LangGraph node/edge definitions
├── main.py                 # Command line chatbot loop
├── requirements.txt
├── pdfs/                   # Place your PDFs to ingest here
│   └── (your files)
└── README.md
```

---

## Notes

- On first run (or when new PDFs are added), ingestion can take time.
- Vector DB is persisted as `chroma_db/`.
- Everything operates locally except Gemini API calls.