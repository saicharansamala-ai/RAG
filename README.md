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
    └── your-file.pdf
```

> **Note:** The `data/` folder is not included in this repo. Create it and add your own PDF before running `create_index.py`.

## Setup

1. Clone the repository

   ```bash
   git clone https://github.com/saicharansamala-ai/RAG.git
   cd RAG
   ```

2. Create a virtual environment and install dependencies

   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # on Windows
   source .venv/bin/activate   # on macOS/Linux

   pip install -r requirements.txt
   ```

3. Add your Groq API key

   Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your-groq-api-key-here
   ```

## Usage

1. Add your PDF

   Create a `data/` folder in the project root and place your PDF inside it (e.g. `data/your-file.pdf`).

2. Build the vector index from your PDF(s) in `data/`

   ```bash
   python create_index.py
   ```

3. Run the chatbot

   ```bash
   python app.py
   ```

4. Ask a question when prompted, and the app will retrieve relevant chunks from the PDF and generate an answer using Groq, along with the source passages it used.

## Example

```text
Ask a question: What is the main contribution of the paper?

Answer: The paper introduces the Transformer, a model architecture based
entirely on attention mechanisms, removing the need for recurrence and
convolutions in sequence transduction tasks.

Sources:
- Abc.pdf, page 1
- Abc.pdf, page 2
```

## Notes

- Swap in a different PDF anytime by replacing the file in `data/` and re-running `create_index.py`.
- Adjust the similarity threshold in the retrieval step to control how strict the context matching is.
