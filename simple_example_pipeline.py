#!/usr/bin/env python3
"""
LangChain OCR FAISS Pipeline
Reads image, performs OCR, creates vector store with retriever

To Run:
pip install langchain langchain-community pillow pytesseract sentence-transformers faiss-cpu
"""

import os
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def setup_tesseract():
    """Configure Tesseract path if needed (Windows)"""
    # Uncomment and modify if on Windows
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\nijjohnson\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    print("Tesseract OCR configured successfully")
    pass

def extract_text_from_image(image_path):
    """Extract text from image using Tesseract OCR"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        print(f"Extracted text length: {len(text)} characters")
        return text
    except Exception as e:
        print(f"Error reading image: {e}")
        return ""

def create_vector_store(text, chunk_size=500, chunk_overlap=50):
    """Create FAISS vector store from text"""
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # Create documents
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    print(f"Created {len(documents)} document chunks")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("Vector store created successfully")
    
    return vectorstore

def test_retrieval(vectorstore, query, k=3):
    """Test the retrieval functionality"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    print(f"\nQuerying: '{query}'")
    results = retriever.get_relevant_documents(query)
    
    print(f"Found {len(results)} relevant documents:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    
    return results

def main():
    # Configuration
    IMAGE_PATH = r"images\unnamed.jpg"  # Change to your image path
    VECTOR_STORE_PATH = "faiss_index"
    
    print("=== LangChain OCR FAISS Pipeline ===\n")
    
    # Setup
    setup_tesseract()
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        print("Please update IMAGE_PATH variable with a valid image file")
        return
    
    # Extract text from image
    print("1. Extracting text from image...")
    text = extract_text_from_image(IMAGE_PATH)
    
    if not text.strip():
        print("No text extracted from image. Check image quality and Tesseract installation.")
        return

    # Write extracted text to a .txt file
    extracted_text_path = os.path.splitext(IMAGE_PATH)[0] + "_extracted.txt"
    with open(extracted_text_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted text written to: {extracted_text_path}")

    print(f"Sample extracted text: {text}...")
    """ 
    # Create vector store
    print("\n2. Creating vector store...")
    vectorstore = create_vector_store(text)
    
    # Save vector store
    print(f"\n3. Saving vector store to {VECTOR_STORE_PATH}...")
    vectorstore.save_local(VECTOR_STORE_PATH)

    # Test retrieval
    print("\n4. Testing retrieval...")
    test_queries = [
        "main topic",
        "important information",
        "key points"
    ]
    
    for query in test_queries:
        test_retrieval(vectorstore, query, k=2)
    
    print(f"\n=== Pipeline Complete ===")
    print(f"Vector store saved to: {VECTOR_STORE_PATH}")
    """

def load_existing_vectorstore(path):
    """Load existing vector store for testing"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    # Run main pipeline
    main()
    
    # Example: Load existing vector store
    # vectorstore = load_existing_vectorstore("faiss_index")
    # test_retrieval(vectorstore, "your query here")