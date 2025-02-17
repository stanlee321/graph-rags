# basic_rag_setup.py
import chromadb


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


data_path = "../data/"

# Initialize a persistent ChromaDB client and create a collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.create_collection("documents")

# Create the vector store using the collection
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load documents from a directory (adjust the path as needed)
documents = SimpleDirectoryReader(data_path).load_data()

# Define an embedding model (using a publicly available model)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Build the index from documents
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

# Create a query engine and run a test query
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic of these documents?")
print(response)
