import os
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
from typing import Dict, List
from dotenv import load_dotenv
from vespa.io import VespaQueryResponse

from libs.pipeline import DocumentRetrievalPipeline  # Ensure this module is in your PYTHONPATH


def float_query_token_vectors(vectors: torch.Tensor) -> Dict[str, List[float]]:
    """Convert query vectors to Vespa format"""
    vespa_token_dict = {}
    for index in range(len(vectors)):
        vespa_token_dict[str(index)] = vectors[index].tolist()
    return vespa_token_dict


def query_vespa(query_text: str, pipeline: DocumentRetrievalPipeline, app, hits: int = 3) -> VespaQueryResponse:
    """Process query and search Vespa"""
    # Get query embeddings using the processor
    dataloader = DataLoader(
        [query_text],
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: pipeline.processor.process_queries(x),
        num_workers=2
    )
    query_embeddings = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(pipeline.model.device) for k, v in batch_query.items()}
            embeddings = pipeline.model(**batch_query)
            if pipeline.model.device == "cuda":
                embeddings = embeddings.float()
            query_embeddings.extend(list(torch.unbind(embeddings.cpu())))
    
    response = app.query(
        yql="select title, url, images from doc where userInput(@userQuery)",
        ranking="default",
        userQuery=query_text,
        timeout=2,
        hits=hits,
        body={
            "presentation.format.tensors": "short-value",
            "format.matchfeatures": "true",
            "input.query(qt)": float_query_token_vectors(query_embeddings[0])
        }
    )
    return response


def scale_image(image: Image.Image, max_width: int) -> Image.Image:
    """Resize image maintaining aspect ratio if width exceeds max_width"""
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        # Use LANCZOS instead of ANTIALIAS in recent Pillow versions
        return image.resize(new_size, Image.LANCZOS)
    return image


def get_best_matching_pages(hit: Dict, num_pages: int = 2) -> List[tuple]:
    """
    Returns a list of tuples (page, score) for the best matching pages.
    If no matchfeatures exist, returns an empty list.
    """
    fields = hit.get("fields", {})
    if "matchfeatures" in fields and "max_sim_per_page" in fields["matchfeatures"]:
        match_scores = fields["matchfeatures"]["max_sim_per_page"]
        # Sort pages by score (descending)
        sorted_pages = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Find the two best num_pages pages
        return [(int(page), score) for page, score in sorted_pages[:num_pages]]
    return []


def print_best_matching_pages(hit: Dict, num_pages: int = 2):
    """Prints to the console the best matching page messages for a given hit."""
    best_pages = get_best_matching_pages(hit, num_pages=num_pages)
    for page, score in best_pages:
        print(f"Best Matching Page {page+1} for PDF document: with MaxSim score {score:.2f}")


def display_query_results_matplotlib(query: str, response: VespaQueryResponse, num_pages: int = 2):
    """
    Displays the query result using matplotlib. For each hit, displays the document title,
    URL, and the best matching pages (with page number and similarity score).
    """
    for i, hit in enumerate(response.hits[:num_pages]):  # Adjust to show more hits if needed
        title = hit["fields"].get("title", "No Title")
        url = hit["fields"].get("url", "")
        images = hit["fields"].get("images", [])
        
        best_pages = get_best_matching_pages(hit)
        # If no matchfeatures are present, fall back to using first image(s)
        if not best_pages and images:
            best_pages = [(idx, None) for idx in range(min(2, len(images)))]
        
        fig, axes = plt.subplots(1, len(best_pages), figsize=(8 * len(best_pages), 8))
        if len(best_pages) == 1:
            axes = [axes]
        fig.suptitle(f"PDF Result {i+1}: {title}\nURL: {url}", fontsize=16)
        
        for ax, (page, score) in zip(axes, best_pages):
            if page < len(images):
                try:
                    # images stored as hex string; convert back to bytes
                    image_data = bytes.fromhex(images[page])
                    image = Image.open(BytesIO(image_data))
                    scaled_image = scale_image(image, 648)
                    ax.imshow(scaled_image)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error loading image:\n{e}", horizontalalignment="center")
            else:
                ax.text(0.5, 0.5, "No image", horizontalalignment="center")
            ax.axis("off")
            if score is not None:
                ax.set_title(f"Best Matching Page {page+1}\nMaxSim Score: {score:.2f}", fontsize=14)
            else:
                ax.set_title(f"Page {page+1}", fontsize=14)
        plt.show()


def run_batch_queries(app, pipeline: DocumentRetrievalPipeline, queries: List[str], num_results: int = 6):
    """
    Runs a list of queries and displays results for each query.
    Prints the basic details along with the best matching page messages,
    then displays the results with matplotlib.
    """
    for query in queries:
        print(f"\nRunning query: {query}")
        response: VespaQueryResponse = query_vespa(query, pipeline, app, num_results)
        print(f"Console Output with total hits {len(response.hits)}:")
        for i, hit in enumerate(response.hits):
            print(f"**********************************START:{i+1} **********************************")
            title = hit['fields'].get('title', 'N/A')
            url = hit['fields'].get('url', 'N/A')
            relevance = hit.get('relevance', 'N/A')
            
            print(f"Title: {title}")
            print(f"URL: {url}")
            print(f"Score: {relevance}")

            print("---")
            # Print best matching pages message
            print_best_matching_pages(hit)
            print(f"**********************************END:{i+1}**********************************")

            
        display_query_results_matplotlib(query, response, 7)


def main():
    load_dotenv()  # load environment variables from .env
    tenant_name = os.getenv("VESPA_TENANT_NAME")
    app_name = os.getenv("VESPA_APP_NAME")
    if not tenant_name or not app_name:
        raise ValueError("VESPA_TENANT_NAME and VESPA_APP_NAME must be set in .env")
    
    pipeline = DocumentRetrievalPipeline()
    
    # Initialize connection to existing Vespa deployment
    app = pipeline.initialize_vespa_app(
        tenant_name=tenant_name,
        app_name=app_name
    )
    
    # Define a list of queries to run
    queries = [
        "What are Agents?"
    ]
    
    run_batch_queries(app, pipeline, queries)


if __name__ == "__main__":
    main()
