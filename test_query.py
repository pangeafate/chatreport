# test_query.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize the embeddings with the same model used for embedding.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the FAISS index saved earlier, enabling dangerous deserialization.
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Define a query to test retrieval.
query = "What are the financial highlights?"
results = vectorstore.similarity_search(query, k=2)  # Retrieve the top 2 similar documents

# Print the results
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content[:300])  # Print first 300 characters of each result
