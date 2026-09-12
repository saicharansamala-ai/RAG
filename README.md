# Basic RAG PDF Chatbot

A beginner-friendly Retrieval-Augmented Generation (RAG) application built with Python, LangChain, FAISS, Hugging Face embeddings, and Groq.

## Architecture

PDF
↓
Document Loading
↓
Text Chunking
↓
Embeddings
↓
FAISS Vector Store
↓
Semantic Retrieval
↓
Similarity Threshold
↓
Context
↓
Groq LLM
↓
Answer + Sources

## Technologies

- Python
- LangChain
- FAISS
- Hugging Face Sentence Transformers
- Groq
- PyPDF
- python-dotenv

## Embedding Model

`sentence-transformers/all-MiniLM-L6-v2`

The model generates 384-dimensional embeddings.

## LLM

Groq:

`openai/gpt-oss-20b`

## Project Structure

```text
Basic-RAG/
│
├── app.py
├── create_index.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    └── Attention.pdf