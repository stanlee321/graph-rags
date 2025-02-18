from libs.pipeline import DocumentRetrievalPipeline
from typing import List, Dict
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize(text: str) -> str:
    # Allow newline and tab; remove other control characters (i.e. code points < 32)
    return ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\t')



def sanitize_document(doc: Dict[str, str]) -> Dict[str, str]:
    sanitized = {}
    for key, value in doc.items():
        if isinstance(value, str):
            sanitized[key] = sanitize(value)
        else:
            sanitized[key] = value
    return sanitized


def main(documents: List[Dict]):
    # Load environment variables
    load_dotenv()
    
    tenant_name = os.getenv("VESPA_TENANT_NAME")
    app_name = os.getenv("VESPA_APP_NAME")
    
    if not tenant_name or not app_name:
        raise ValueError("VESPA_TENANT_NAME and VESPA_APP_NAME must be set in .env file")

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = DocumentRetrievalPipeline()
    
    # Process documents
    logger.info("Processing documents...")
    vespa_feed = pipeline.process_documents(documents)
    
    # Deploy to Vespa Cloud
    logger.info("Deploying to Vespa Cloud...")
    app = pipeline.deploy_to_vespa(
        tenant_name=tenant_name,
        app_name=app_name
    )
    
    # Feed documents to Vespa
    logger.info("Feeding documents to Vespa...")
    with app.syncio() as sync:
        for operation in vespa_feed:
            fields = sanitize_document(operation["fields"])
            response = sync.feed_data_point(
                data_id=fields["url"],
                fields=fields,
                schema="doc"
            )
            if not response.is_successful():
                logger.error(f"Error feeding document: {response.json()}")
            else:
                logger.info(f"Successfully fed document: {operation['fields']['title']}")

if __name__ == "__main__":
    # Sample documents
    documents = [
        # {
        #     "title": "ColBERTv2: Effective and Efficient Retrieval",
        #     "url": "https://arxiv.org/pdf/2112.01488.pdf"
        # },
        # {
        #     "title": "ColBERT: Efficient and Effective Passage Search",
        #     "url": "https://arxiv.org/pdf/2004.12832.pdf"
        # }
        {
            "title": "KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment",
            "url": "https://arxiv.org/pdf/2502.06472.pdf"
        }
    ]
    
    main(documents)

