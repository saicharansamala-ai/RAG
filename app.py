from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Load embedding model
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ============================================================
# 3. Load existing FAISS vector store
# ============================================================

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS vector store loaded")


# ============================================================
# 4. Create Groq LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)


# ============================================================
# 5. Create prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the context provided below.

    If the answer is not present in the context, say:
    "I don't know based on the provided document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


# ============================================================
# 6. Create LangChain chain
# ============================================================

chain = prompt | llm | StrOutputParser()


# ============================================================
# 7. Interactive RAG chatbot
# ============================================================

while True:

    query = input("\nYou: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break


    # ========================================================
    # 8. Retrieve relevant chunks
    # ========================================================

    results = vectorstore.similarity_search_with_score(
        query,
        k=3
    )


    # ========================================================
    # 9. Apply similarity threshold
    # ========================================================

    threshold = 1.2

    results = [
        (result, score)
        for result, score in results
        if score <= threshold
    ]


    # ========================================================
    # 10. Check if relevant chunks were found
    # ========================================================

    if not results:

        print("\nFinal Answer:")
        print(
            "I don't know based on the provided document."
        )

        continue


    # ========================================================
    # 11. Create context
    # ========================================================

    context = "\n\n".join(
        f"[Page {result.metadata.get('page', 0) + 1}]\n"
        f"{result.page_content}"
        for result, score in results
    )


    # ========================================================
    # 12. Send context + question to LLM
    # ========================================================

    answer = chain.invoke(
        {
            "context": context,
            "question": query
        }
    )


    # ========================================================
    # 13. Display final answer
    # ========================================================

    print("\nFinal Answer:")
    print(answer)


    # ========================================================
    # 14. Display sources
    # ========================================================

    print("\nSources:")

    for result, score in results:

        page = result.metadata.get("page", 0) + 1

        print(
            f"- Attention.pdf — Page {page}"
        )