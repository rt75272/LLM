from ingest import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

DB_DIR = "./chroma_db"
DATA_DIR = "data"

def create_vector_db():
    # 1. Load the raw documents from Day 1
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found. Exiting.")
        return

    # 2. Initialize the text splitter
    # We use a 1000-character chunk with a 200-character overlap to preserve context between chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Initialize the open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and persist the Chroma vector database
    print("Building Chroma vector database. This may take a moment...")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print(f"Database successfully built and saved to {DB_DIR}/")

if __name__ == "__main__":
    create_vector_db()
