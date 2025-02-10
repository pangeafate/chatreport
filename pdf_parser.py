# pdf_parser.py
import os
from unstructured.partition.pdf import partition_pdf
from langchain.schema import Document

def parse_pdf_to_text(file_path):
    """
    Uses the Unstructured API to parse a PDF file and return its text content.
    """
    elements = partition_pdf(filename=file_path)
    # Combine all text elements. We check if each element has a 'text' attribute.
    text = "\n".join([element.text for element in elements if hasattr(element, "text") and element.text])
    return text

def create_document_from_pdf(file_path):
    """
    Parses a PDF and converts the result into a LangChain Document.
    """
    text = parse_pdf_to_text(file_path)
    filename = os.path.basename(file_path)
    # Create a LangChain Document with the parsed text and metadata
    doc = Document(page_content=text, metadata={"source": filename})
    return doc
