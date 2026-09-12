from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. Load PDF
# ============================================================

loader = PyPDFLoader("data/Attention.pdf")

documents = loader.load()

print("Number of pages:", len(documents))


# ============================================================
# 2. Split PDF into chunks
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# ============================================================
# 3. Create embedding model
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ============================================================
# 4. Create FAISS vector store
# ============================================================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created")


# ============================================================
# 5. Save FAISS vector store
# ============================================================

vectorstore.save_local("faiss_index")

print("FAISS vector store saved")