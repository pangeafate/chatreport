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

# Load the FAISS index using the same embeddings used for creation.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Create a RetrievalQA chain using a chain type that can process context properly.
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0, max_tokens=500),
    chain_type="stuff",  # Changed from "staff" to "stuff"
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
)

def answer_question(question: str) -> tuple[str, list[str]]:
    """
    Given a question, retrieve relevant documents, extract source file names,
    and return the answer along with a list of source files.
    """
    # Retrieve the relevant documents from the vectorstore.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    relevant_docs = retriever.get_relevant_documents(question)
    
    # Extract unique source file names from the metadata.
    sources = list({doc.metadata.get("source", "unknown") for doc in relevant_docs})
    
    # Generate the answer using the QA chain.
    answer = qa_chain.run(question)
    return answer, sources
