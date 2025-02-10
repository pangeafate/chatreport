# debug_print.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Retrieve a few documents for a sample query (or just inspect all documents)
docs = vectorstore.as_retriever(search_kwargs={"k": 3}).get_relevant_documents("financial highlights")
for i, doc in enumerate(docs):
    print(f"Document {i+1} (first 300 characters):")
    print(doc.page_content[:300])
    print("-" * 50)