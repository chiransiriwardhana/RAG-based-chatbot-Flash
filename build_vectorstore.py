from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import shutil
import os

PDF_FILES = [
    "globalwarming.pdf",
    "/Users/chiransiriwardena/Documents/Rag_chatbot/pdf/pop.pdf"
]

# Reset DB when documents change
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")

docs = []
for path in PDF_FILES:
    docs.extend(PyPDFLoader(path).load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="llama3.2")

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("✅ Vectorstore built successfully")