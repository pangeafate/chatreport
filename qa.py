# qa.py
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI  # Use ChatOpenAI for chat models

# Ensure your OpenAI API key is available.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

# Use the same environment variable as embed.py for the FAISS index path.
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/data/uploads/faiss_index")

# Check if the FAISS index file exists; the expected file is "index.faiss" in FAISS_INDEX_PATH.
index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
if not os.path.exists(index_file):
    print(f"[INFO] FAISS index not found at {index_file}. Running embedding process...")
    try:
        from embed import embed_documents
        embed_documents()  # Run the embedding process synchronously.
    except Exception as e:
        raise RuntimeError(f"Embedding process failed: {e}")
    # Verify that the index file was created.
    if not os.path.exists(index_file):
        raise RuntimeError(f"Embedding process completed, but FAISS index file still not found at {index_file}.")
    else:
        print(f"[INFO] FAISS index successfully created at {index_file}.")

# Initialize embeddings and load the FAISS index.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Create a RetrievalQA chain using a chain type that can process context properly.
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0, max_tokens=500),
    chain_type="stuff",  # Ensure you use the appropriate chain type.
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
)

def answer_question(question: str) -> tuple[str, list[str]]:
    """
    Given a question, retrieve relevant documents, extract source file names,
    and return the answer along with a list of source files.
    """
    # Retrieve the relevant documents from the vectorstore.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(question)
    
    # Extract unique source file names from the metadata.
    sources = list({doc.metadata.get("source", "unknown") for doc in relevant_docs})
    
    # Generate the answer using the QA chain.
    answer = qa_chain.run(question)
    return answer, sources
