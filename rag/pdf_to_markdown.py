# pdf_to_markdown.py
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Optional: Provide configuration to paginate the output so page numbers are preserved.
config = {"paginate_output": True}

# Initialize the converter with the model artifacts and configuration.
converter = PdfConverter(
    artifact_dict=create_model_dict(),
    config=config  # This instructs Marker to include pagination markers.
)

# Convert the PDF file to Markdown
pdf_path = "../data/2502.06472v1.pdf"
rendered = converter(pdf_path)

# Extract markdown text, metadata, and images from the rendered output.
markdown_text, metadata, images = text_from_rendered(rendered)

# Print the converted Markdown and associated metadata
print("Markdown Output:\n", markdown_text)
print("\nMetadata:\n", metadata)