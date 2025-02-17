import torch
from torch.utils.data import DataLoader
from typing import Dict, List
from libs.pipeline import DocumentRetrievalPipeline

def float_query_token_vectors(vectors: torch.Tensor) -> Dict[str, List[float]]:
    """Convert query vectors to Vespa format"""
    vespa_token_dict = {}
    for index in range(len(vectors)):
        vespa_token_dict[str(index)] = vectors[index].tolist()
    return vespa_token_dict

def query_vespa(query_text: str, pipeline: DocumentRetrievalPipeline, app):
    """Process query and search Vespa"""
    # Get query embeddings
    dataloader = DataLoader(
        [query_text],
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: pipeline.processor.process_queries(x)
    )
    
    query_embeddings = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(pipeline.model.device) for k, v in batch_query.items()}
            embeddings = pipeline.model(**batch_query)
            if pipeline.model.device == "cuda":
                embeddings = embeddings.float()
            query_embeddings.extend(list(torch.unbind(embeddings.cpu())))

    # Query Vespa
    response = app.query(
        yql="select title, url, images, matchfeatures.* from doc where userInput(@userQuery)",
        ranking="default",
        userQuery=query_text,
        timeout=2,
        hits=3,
        body={
            "presentation.format.tensors": "short-value",
            "input.query(qt)": float_query_token_vectors(query_embeddings[0])
        }
    )
    return response

def main(query_text: str):
    pipeline = DocumentRetrievalPipeline()
    
    # Initialize connection to existing Vespa deployment
    app = pipeline.initialize_vespa_app(
        tenant_name="my-tenant-name",
        app_name="visionrag"
    )
    
    # Query Vespa
    response = query_vespa(query_text, pipeline, app)
    
    print("\nQuery Results:")
    for hit in response.hits:
        print(f"Title: {hit['fields']['title']}")
        print(f"URL: {hit['fields']['url']}")
        print(f"Score: {hit['relevance']}")
        if 'matchfeatures' in hit['fields']:
            print(f"Match features: {hit['fields']['matchfeatures']}")
        print("---")

if __name__ == "__main__":
    main("Composition of the LoTTE benchmark")