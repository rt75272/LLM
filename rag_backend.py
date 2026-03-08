from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DB_DIR = "./chroma_db"

def get_rag_chain():
    # 1. Re-initialize the embedding model and connect to the existing database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # 2. Create the retriever (fetches the top 3 most relevant chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. Initialize the local LLM via Ollama
    llm = ChatOllama(model="mistral")

    # 4. Define the prompt template
    template = """Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Keep the answer concise.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    prompt = PromptTemplate.from_template(template)

    # Helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Construct the LangChain Expression Language (LCEL) chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    # Quick test in the terminal
    chain = get_rag_chain()
    test_question = "What is the main topic of the documents provided?"
    print(f"Question: {test_question}\n")
    print("Thinking...\n")
    print(chain.invoke(test_question))
