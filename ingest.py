import os
from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(data_dir: str):
    """
    Loads all PDF documents from the specified directory.
    """
    print(f"Scanning directory: {data_dir} for PDFs...")
    
    # Check if directory exists and has files
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Error: The directory '{data_dir}' is empty or does not exist.")
        return []

    # Initialize the LangChain PDF directory loader
    loader = PyPDFDirectoryLoader(data_dir)
    
    # Load and parse the PDFs
    documents = loader.load()
    
    print(f"Successfully loaded {len(documents)} pages across all PDFs.")
    return documents

if __name__ == "__main__":
    # Define the path to our data folder
    DIRECTORY_PATH = "data"
    
    # Run the loader
    docs = load_documents(DIRECTORY_PATH)
    
    # Print a sample of the first document to verify it worked
    if docs:
        print("\n--- Sample Output of First Page ---")
        print(docs[0].page_content[:500] + "...\n")
        print(f"Source metadata: {docs[0].metadata}")
