# main.py

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from llama_index.core import Document, VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

def convert_pdf_to_markdown(pdf_path):
    # Enable pagination to include page numbers in the output.
    config = {"paginate_output": True}
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config=config
    )
    rendered = converter(pdf_path)
    markdown_text, metadata, images = text_from_rendered(rendered)
    return markdown_text, metadata

def build_rag_index(documents, embed_model_name="BAAI/bge-base-en-v1.5"):
    # Initialize persistent ChromaDB client and create or get a collection.
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    # For simplicity, we create a new collection each run.
    chroma_collection = chroma_client.create_collection("documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Initialize embedding model.
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    
    # Build the index from provided documents.
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    return index

def main():
    # Specify the path to the PDF to process.
    pdf_path = "../data/2502.06472v1.pdf"
    
    print("Converting PDF to Markdown...")
    markdown_text, metadata = convert_pdf_to_markdown(pdf_path)
    print("Conversion complete.\n--- Markdown Output ---\n", markdown_text)
    print("\n--- Metadata ---\n", metadata)
    
    # Create a LlamaIndex Document using the Markdown text and its metadata.
    doc = Document(text=markdown_text, extra_info=metadata)
    
    print("\nBuilding RAG index with the converted document...")
    index = build_rag_index([doc])
    
    # Create a query engine from the index.
    query_engine = index.as_query_engine()
    
    # Example query to test page information (assuming the conversion preserved page markers)
    test_query = "What information is provided on page 3?"
    print("\nRunning query:", test_query)
    response = query_engine.query(test_query)
    print("\nQuery response:\n", response)

if __name__ == "__main__":
    main()