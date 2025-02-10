# embed.py
import os
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Use environment variables for persistent storage paths.
# On Render, you can mount your persistent disk at /data/uploads.
# If no environment variable is set, the defaults will use the Render mount path.
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/data/uploads")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/data/uploads/faiss_index")

def parse_pdf_to_text(file_path: str) -> str:
    """
    Parses the PDF file using unstructured.partition.pdf and returns the combined text.
    """
    print(f"[DEBUG] Parsing PDF file: {file_path}")
    elements = partition_pdf(filename=file_path)
    # Join together text from elements that have a text attribute.
    text = "\n".join(
        [element.text for element in elements if hasattr(element, "text") and element.text]
    )
    print(f"[DEBUG] Extracted text length: {len(text)} characters")
    return text

def create_documents_from_pdf(file_path: str) -> list:
    """
    Parses a PDF and splits the text into smaller chunks.
    Returns a list of Document objects.
    """
    text = parse_pdf_to_text(file_path)
    # Create a text splitter with a chunk size of 3000 characters and an overlap of 300.
    print(f"[DEBUG] Splitting text from {file_path} into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    texts = text_splitter.split_text(text)
    documents = [
        Document(page_content=t, metadata={"source": os.path.basename(file_path)})
        for t in texts
    ]
    print(f"[DEBUG] Created {len(documents)} document chunks from {os.path.basename(file_path)}")
    return documents

def embed_documents():
    # Use the persistent uploads folder path.
    uploads_folder = UPLOAD_FOLDER

    # Ensure the uploads folder exists.
    if not os.path.exists(uploads_folder):
        print(f"[DEBUG] The folder '{uploads_folder}' does not exist. Creating it.")
        os.makedirs(uploads_folder)
    else:
        print(f"[DEBUG] Found uploads folder: '{uploads_folder}'")

    all_documents = []

    # List all PDF files in the uploads folder.
    pdf_files = [file for file in os.listdir(uploads_folder) if file.lower().endswith(".pdf")]
    print(f"[DEBUG] Found {len(pdf_files)} PDF file(s) in '{uploads_folder}': {pdf_files}")

    # Process each PDF file.
    for file in pdf_files:
        file_path = os.path.join(uploads_folder, file)
        print(f"[DEBUG] Processing file: {file_path}")
        try:
            docs = create_documents_from_pdf(file_path)
            all_documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Error processing {file}: {e}")

    if not all_documents:
        print("[DEBUG] No PDF documents found for embedding.")
        return

    # Initialize embeddings.
    print("[DEBUG] Initializing embeddings using model 'all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the FAISS vector store from the documents.
    print("[DEBUG] Creating FAISS vector store from document chunks...")
    vectorstore = FAISS.from_documents(all_documents, embeddings)

    # Save the FAISS index locally using the persistent path.
    print(f"[DEBUG] Saving FAISS index to the '{FAISS_INDEX_PATH}' folder...")
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"[DEBUG] FAISS index saved to the '{FAISS_INDEX_PATH}' folder.")

if __name__ == "__main__":
    embed_documents()
