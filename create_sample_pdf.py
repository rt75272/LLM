from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def create_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    text = """
    Enterprise RAG Assistant Documentation
    
    Overview
    This project is a local RAG assistant that allows users to query a knowledge base of PDF documents using a local LLM.
    
    Architecture
    1. Ingestion: Parses text from PDFs in the data/ directory.
    2. Embeddings: Converts text chunks into vector embeddings using all-MiniLM-L6-v2.
    3. Vector Store: Stores embeddings in a local ChromaDB.
    4. Retrieval: Searches for relevant context based on user queries.
    5. Generation: Uses a local LLM (Mistral via Ollama) to generate answers.
    6. Interface: Provides a chat interface using Streamlit.
    
    Setup
    1. Install Dependencies: pip install -r requirements.txt (or use uv sync)
    2. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
    3. Pull Model: ollama pull mistral
    4. Prepare Data: Place PDFs in data/
    5. Build Database: python build_db.py
    6. Run Application: streamlit run app.py
    
    This is a sample document for testing the RAG system.
    """
    
    y = 750
    for line in text.split('\n'):
        c.drawString(50, y, line.strip())
        y -= 20
        
    c.save()
    print(f"Created {filename}")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    create_pdf("data/sample.pdf")
