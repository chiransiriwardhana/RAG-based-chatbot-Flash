from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

PDF_FILES = [
    "globalwarming.pdf",
    # Add more PDF paths as needed
]

def load_vectorstore(pdf_paths=None):
    if pdf_paths is None:
        pdf_paths = PDF_FILES
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="llama3.2")
    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_db"
    )

def retrieve_context(vectorstore, query: str) -> str:
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(d.page_content for d in docs) if docs else ""
