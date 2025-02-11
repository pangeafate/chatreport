import os
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Use environment variables for persistent storage paths.
# On Render, mount your persistent disk at /data/uploads.
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
    print(f"[DEBUG] Splitting text from {file_path} into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    documents = [
        Document(page_content=t, metadata={"source": os.path.basename(file_path)})
        for t in texts
    ]
    print(f"[DEBUG] Created {len(documents)} document chunks from {os.path.basename(file_path)}")
    return documents

def embed_documents():
    uploads_folder = UPLOAD_FOLDER

    if not os.path.exists(uploads_folder):
        print(f"[DEBUG] The folder '{uploads_folder}' does not exist. Creating it.")
        os.makedirs(uploads_folder, exist_ok=True)
    else:
        print(f"[DEBUG] Found uploads folder: '{uploads_folder}'")

    # Define the tracking file to store processed PDF filenames.
    processed_files_path = os.path.join(uploads_folder, "processed_files.txt")
    processed_files = []
    if os.path.exists(processed_files_path):
        with open(processed_files_path, "r") as f:
            processed_files = [line.strip() for line in f.readlines() if line.strip()]
    print(f"[DEBUG] Already processed files: {processed_files}")

    # Check if the FAISS index exists; if not, reset processed_files so all PDFs are processed.
    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    if not os.path.exists(index_file):
        print(f"[DEBUG] FAISS index file not found at {index_file}. Rebuilding index from scratch.")
        processed_files = []

    # List all PDFs in the uploads folder.
    all_pdf_files = [file for file in os.listdir(uploads_folder) if file.lower().endswith(".pdf")]
    # Filter out PDFs that have already been processed.
    new_pdf_files = [file for file in all_pdf_files if file not in processed_files]
    print(f"[DEBUG] Found {len(new_pdf_files)} new PDF file(s) in '{uploads_folder}': {new_pdf_files}")

    new_documents = []
    # Process each new PDF file.
    for file in new_pdf_files:
        file_path = os.path.join(uploads_folder, file)
        print(f"[DEBUG] Processing new file: {file_path}")
        try:
            docs = create_documents_from_pdf(file_path)
            new_documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Error processing {file}: {e}")

    if not new_documents:
        print("[DEBUG] No new document chunks found for embedding.")
        return

    print("[DEBUG] Initializing embeddings using model 'all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if an existing FAISS index exists.
    if os.path.exists(index_file):
        print("[DEBUG] Loading existing FAISS index...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("[DEBUG] Adding new documents to the existing FAISS index...")
        vectorstore.add_documents(new_documents)
    else:
        print("[DEBUG] No existing FAISS index found. Creating a new one from new documents...")
        vectorstore = FAISS.from_documents(new_documents, embeddings)

    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"[DEBUG] The FAISS index directory '{FAISS_INDEX_PATH}' does not exist. Creating it.")
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    print(f"[DEBUG] Saving FAISS index to the '{FAISS_INDEX_PATH}' folder...")
    vectorstore.save_local(FAISS_INDEX_PATH)

    # Update the tracking file with the newly processed PDFs.
    with open(processed_files_path, "a") as f:
        for file in new_pdf_files:
            f.write(file + "\n")
    print(f"[DEBUG] Updated processed files list saved to {processed_files_path}")

    try:
        files = os.listdir(FAISS_INDEX_PATH)
        print(f"[DEBUG] Files in FAISS index directory: {files}")
    except Exception as e:
        print(f"[ERROR] Could not list FAISS index directory: {e}")

    print(f"[DEBUG] FAISS index updated and saved to the '{FAISS_INDEX_PATH}' folder.")

if __name__ == "__main__":
    embed_documents()
