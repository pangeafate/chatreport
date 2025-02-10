# embed.py
import os
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def parse_pdf_to_text(file_path: str) -> str:
    """
    Parses the PDF file using unstructured.partition.pdf and returns the combined text.
    """
    elements = partition_pdf(filename=file_path)
    # Join together text from elements that have a text attribute
    text = "\n".join([element.text for element in elements if hasattr(element, "text") and element.text])
    return text

def create_documents_from_pdf(file_path: str) -> list:
    """
    Parses a PDF and splits the text into smaller chunks.
    Returns a list of Document objects.
    """
    text = parse_pdf_to_text(file_path)
    # Create a text splitter with a chunk size of 500 characters and an overlap of 50.
    # (Adjust these values as needed; if you prefer token counts, you might convert token counts to approximate character counts.)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    # Create a Document for each chunk, tagging it with the source file name.
    documents = [Document(page_content=t, metadata={"source": os.path.basename(file_path)}) for t in texts]
    return documents

def embed_documents():
    uploads_folder = "uploads"
    all_documents = []
    
    # Process each PDF file in the uploads folder.
    for file in os.listdir(uploads_folder):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(uploads_folder, file)
            print(f"Processing {file}...")
            try:
                docs = create_documents_from_pdf(file_path)
                print(f"Created {len(docs)} document chunks from {file}.")
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    if not all_documents:
        print("No documents found for embedding.")
        return

    # Initialize embeddings using a Sentence Transformers model via HuggingFaceEmbeddings.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the FAISS vector store from the documents.
    vectorstore = FAISS.from_documents(all_documents, embeddings)
    
    # Save the FAISS index locally.
    vectorstore.save_local("faiss_index")
    print("FAISS index saved to the 'faiss_index' folder.")

if __name__ == "__main__":
    embed_documents()
