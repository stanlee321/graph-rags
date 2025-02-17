# main.py

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import os

# Set the API key as an environment variable
os.environ["GROQ_API_KEY"] = "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"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def convert_pdf_to_markdown(pdf_path):
    config = {"paginate_output": True}
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config=config
    )
    rendered = converter(pdf_path)
    markdown_text, metadata, images = text_from_rendered(rendered)
    return markdown_text, metadata

def build_rag_index(documents):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.create_collection("documents", get_or_create=True)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

def main():
    pdf_path = "../data/2502.06472v1.pdf"
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device="cuda"
    )

    print("Converting PDF to Markdown...")
    markdown_text, metadata = convert_pdf_to_markdown(pdf_path)
    print("Conversion complete.\n--- Markdown Output ---\n", markdown_text)
    print("\n--- Metadata ---\n", metadata)
    
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    
    doc = Document(text=markdown_text, extra_info=metadata)
    print("\nBuilding RAG index with the converted document...")
    index = build_rag_index([doc])
    
    # Create a query engine from the index
    query_engine = index.as_query_engine(llm=llm)
    
    # Chat loop
    print("\nEntering chat mode. Type your questions below (type 'quit' or 'exit' to end):")
    while True:
        user_input = input("Your query: ")
        if user_input.strip().lower() in ["quit", "exit"]:
            print("Exiting chat mode. Goodbye!")
            break
        response = query_engine.query(user_input)
        print("\nResponse:\n", response, "\n")

if __name__ == "__main__":
    main()
