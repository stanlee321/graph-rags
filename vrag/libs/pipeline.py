import os
import torch
import requests
from torch.utils.data import DataLoader

from io import BytesIO
from typing import Dict, List
from tqdm import tqdm
from colpali_engine.utils.torch_utils import get_torch_device
from colpali_engine.models import ColPali, ColPaliProcessor

from vespa.package import Schema, Document, Field, FieldSet, ApplicationPackage
from vespa.deployment import VespaCloud
from vespa.package import RankProfile, Function, FirstPhaseRanking, SecondPhaseRanking

from PyPDF2 import PdfReader
from pdf2image import convert_from_path


def sanitize_text(text: str) -> str:
    # Keep characters with code point >= 32 or newline/tab
    return ''.join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t")


class DocumentRetrievalPipeline:
    def __init__(self, model_name="vidore/colpali-v1.3"): 
        self.setup_device()
        self.setup_model(model_name)
        self.setup_vespa_schema()
        self.key = os.getenv("VESPA_TEAM_API_KEY")

    def setup_device(self):
        """Initialize device and data type settings"""
        self.device = get_torch_device("auto")
        self.dtype = torch.bfloat16 if "cuda" in self.device  else torch.float32

    def setup_model(self, model_name):
        """Initialize ColPali model and processor"""
        self.model = ColPali.from_pretrained(
            model_name, 
            torch_dtype=self.dtype, 
            device_map=self.device
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name, torch_dtype=torch.float16)

    def download_pdf(self, url: str) -> BytesIO:
        """Download PDF from URL"""
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF: Status code {response.status_code}")
        return BytesIO(response.content)

    def process_pdf(self, pdf_url: str):
        """Process PDF to extract images and text"""
        pdf_file = self.download_pdf(pdf_url)
        
        # Save PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())
            
        reader = PdfReader("temp.pdf")
        page_texts = [page.extract_text() for page in reader.pages]
        images = convert_from_path("temp.pdf")
        
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
            
        return images, page_texts

    def generate_embeddings(self, images):
        """Generate embeddings for images using ColPali"""
        page_embeddings = []
        dataloader = DataLoader(
            images,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x)
        )
        
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
                if self.model.device == "cuda":
                    embeddings_doc = embeddings_doc.float()
                page_embeddings.extend(list(torch.unbind(embeddings_doc.cpu())))
                
        return page_embeddings

    def setup_vespa_schema(self):
        """Configure Vespa schema and application"""
        self.schema = Schema(
            name="doc",
            document=Document(
                fields=[
                    Field(name="url", type="string", indexing=["summary"]),
                    Field(
                        name="title",
                        type="string",
                        indexing=["summary", "index"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="texts",
                        type="array<string>",
                        indexing=["index"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="images",
                        type="array<string>",
                        indexing=["summary"],
                    ),
                    Field(
                        name="colbert",
                        type="tensor<int8>(page{}, patch{}, v[16])",
                        indexing=["attribute"],
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["title", "texts"])],
        )
        
        
        colbert_profile = RankProfile(
            name="default",
            inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
            functions=[
                Function(
                    name="max_sim_per_page",
                    expression="""
                        sum(
                            reduce(
                                sum(
                                    query(qt) * unpack_bits(attribute(colbert)) , v
                                ),
                                max, patch
                            ),
                            querytoken
                        )
                    """,
                ),
                Function(name="max_sim", expression="reduce(max_sim_per_page, max, page)"),
                Function(name="bm25_score", expression="bm25(title) + bm25(texts)"),
            ],
            first_phase=FirstPhaseRanking(expression="bm25_score"),
            second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
            match_features=["max_sim_per_page", "bm25_score"],
        )
        self.schema.add_rank_profile(colbert_profile)

    def deploy_to_vespa(self, tenant_name: str, app_name: str):
        """Deploy application to Vespa Cloud"""
        

        app_package = ApplicationPackage(name=app_name, schema=[self.schema])


        if self.key:
            self.key = self.key.replace(r"\n", "\n")
            
        vespa_cloud = VespaCloud(
            tenant=tenant_name,
            application=app_name,
            key_content=self.key,
            application_package=app_package,
        )
        
        return vespa_cloud.deploy()
    
    def initialize_vespa_app(self, tenant_name: str, app_name: str):
        if self.key:
            self.key = self.key.replace(r"\n", "\n")
        # Create the ApplicationPackage using the existing schema
        self.setup_vespa_schema()
        app_package = ApplicationPackage(name=app_name, schema=[self.schema])
        
        vespa_cloud = VespaCloud(
            tenant=tenant_name,
            application=app_name,
            key_content=self.key,
            application_package=app_package  # now provided!
        )
        return vespa_cloud.get_application()  # This connects to the existing deployment


    def process_documents(self, documents: List[Dict]):
        """Process documents and prepare for Vespa indexing"""
        vespa_feed = []
        
        for doc in documents:
            images, texts = self.process_pdf(doc["url"])
            
            sanitized_texts = [sanitize_text(t) for t in texts if t]

            embeddings = self.generate_embeddings(images)

            # Convert images to base64
            images_base64 = [self.image_to_base64(img) for img in images]
            
            vespa_doc = {
                "fields": {
                    "url": doc["url"],
                    "title": doc["title"],
                    "images": images_base64,
                    "texts": sanitized_texts,
                    "colbert": {
                        "blocks": self.binarize_vectors(embeddings)
                    }
                }
            }
            vespa_feed.append(vespa_doc)
            
        return vespa_feed

    @staticmethod
    def image_to_base64(image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return str(buffered.getvalue().hex())

    @staticmethod
    def binarize_vectors(vectors: List[torch.Tensor]) -> Dict[str, str]:
        """Binarize token vectors for Vespa storage"""
        import numpy as np
        from binascii import hexlify
        
        vespa_tensor = []
        for page_id, page_vector in enumerate(vectors):
            binarized = np.packbits(
                np.where(page_vector > 0, 1, 0), 
                axis=1
            ).astype(np.int8)
            
            for patch_index in range(len(page_vector)):
                values = str(hexlify(binarized[patch_index].tobytes()), "utf-8")
                if values == "00000000000000000000000000000000":
                    continue
                    
                vespa_tensor.append({
                    "address": {"page": page_id, "patch": patch_index},
                    "values": values
                })
                
        return vespa_tensor
