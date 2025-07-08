from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(title="RAG System API", description="A simple RAG system using LangChain and local Llama")

host = "https://chatgpt.microsopht.com/ollama"
apiKey = "sk-5cc8dd8ffa24407fa0c0a0c9cad52e18"
model = "llama3:latest"

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list = []

class DocumentRequest(BaseModel):
    text: str
    metadata: dict = {}

# Global variables for the RAG system
llm = None
vectorstore = None
qa_chain = None

# Initialize the RAG system
def initialize_rag():
    global llm, vectorstore, qa_chain
    
    # Initialize Llama model (adjust model name as needed)
    # llm = Ollama(
    #     model="llama2",  # Change this to your local model name
    #     base_url="http://localhost:11434"  # Default Ollama server URL
    # )
    llm = ChatOpenAI(
        openai_api_base=host,  # Replace with your Llama server URL
        openai_api_key=apiKey,  # Replace with your API key if needed
        model=model,        # Replace with your model name
        temperature=0.2
    )
    
    # Initialize embeddings
    # embeddings = OllamaEmbeddings(
    #     model="llama2",  # Change this to your embedding model
    #     base_url="http://localhost:11434"
    # )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load existing FAISS vectorstore instead of creating a new one
    faiss_path = "faiss_index"  # Change to your actual FAISS directory path
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        print("Loaded existing FAISS vector store.")
    else:
        # Optionally, handle the case where the vector store doesn't exist
        vectorstore = FAISS.from_texts(
            texts=["Initial document"],
            embedding=embeddings,
            metadatas=[{"source": "initial"}]
        )
        print("Created new FAISS vector store.")
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    initialize_rag()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG System API is running"}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system"""
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Query the RAG system
        result = qa_chain({"query": request.question})
        
        # Extract sources from result
        sources = []
        if "source_documents" in result:
            sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        
        return QueryResponse(
            answer=result["result"],
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# @app.post("/add_document")
# async def add_document(request: DocumentRequest):
#     """Add a document to the vector store"""
#     global vectorstore
    
#     if vectorstore is None:
#         raise HTTPException(status_code=500, detail="Vector store not initialized")
    
#     try:
#         # Split text into chunks
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         chunks = text_splitter.split_text(request.text)
        
#         # Add chunks to vector store
#         metadatas = [request.metadata for _ in chunks]
#         vectorstore.add_texts(chunks, metadatas=metadatas)
        
#         return {"message": f"Added {len(chunks)} chunks to the vector store"}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

# @app.post("/add_file")
# async def add_file(file_path: str):
#     """Add a text file to the vector store"""
#     global vectorstore
    
#     if vectorstore is None:
#         raise HTTPException(status_code=500, detail="Vector store not initialized")
    
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")
    
#     try:
#         # Load document
#         loader = TextLoader(file_path)
#         documents = loader.load()
        
#         # Split documents
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         chunks = text_splitter.split_documents(documents)
        
#         # Add to vector store
#         vectorstore.add_documents(chunks)
        
#         return {"message": f"Added {len(chunks)} chunks from file {file_path}"}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error adding file: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "llm_initialized": llm is not None,
        "vectorstore_initialized": vectorstore is not None,
        "qa_chain_initialized": qa_chain is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)