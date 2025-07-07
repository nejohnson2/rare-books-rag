from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load and split documents
loader = TextLoader(r"images\unnamed_extracted.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build vector store
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# Search example
query = "What is said about wood stress?"
results = vectorstore.similarity_search(query, k=5)

for doc in results:
    print(doc.page_content)