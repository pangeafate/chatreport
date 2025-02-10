import os
# Disable tokenizers parallelism to avoid warnings about forking.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
import threading
from embed import embed_documents  # Ensure your embed.py defines embed_documents()

# Import answer_question at the top to avoid circular import issues.
from qa import answer_question

# Use environment variables for persistent storage paths.
# On Render, set UPLOAD_FOLDER and FAISS_INDEX_PATH to your mounted disk paths.
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/data/uploads")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/data/uploads/faiss_index")
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Change this to a secure key in production

def allowed_file(filename):
    """Check if the file has a PDF extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)

        files = request.files.getlist('file')
        for file in files:
            if file.filename == '':
                flash('One of the selected files has no filename.')
                continue
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Ensure the upload folder exists.
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                flash(f'File "{filename}" successfully uploaded.')
            else:
                flash(f'File "{file.filename}" is not a PDF or not allowed.')
        return redirect(url_for('upload_file'))
    return render_template('upload.html')

@app.route('/parse/<path:filename>')
def parse_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return f"File {filename} not found.", 404
    from pdf_parser import create_document_from_pdf
    doc = create_document_from_pdf(file_path)
    preview = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
    return f"<h1>Parsed Content of {filename}</h1><pre>{preview}</pre>"

# Global initialization for embeddings and vectorstore.
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize embeddings.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Ensure the FAISS index directory exists.
if not os.path.exists(FAISS_INDEX_PATH):
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

# Attempt to load the FAISS index.
try:
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    vectorstore = None
    print("Error loading FAISS index:", e)

# Global status for the embedding process.
embedding_status = {"running": False, "message": ""}

def background_embedding():
    global embedding_status, vectorstore, embeddings
    embedding_status["running"] = True
    try:
        embed_documents()  # Run the embedding process (this rebuilds the FAISS index)
        # Reload the vectorstore after embedding using the persistent FAISS index path.
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        embedding_status["message"] = "Embedding successful!"
    except Exception as e:
        embedding_status["message"] = f"Embedding error: {str(e)}"
    finally:
        embedding_status["running"] = False

@app.route('/start_embedding', methods=['POST'])
def start_embedding():
    global embedding_status
    if embedding_status["running"]:
        return {"status": "running", "message": "Embedding is already in progress."}
    thread = threading.Thread(target=background_embedding)
    thread.start()
    return {"status": "started", "message": "Embedding started. Please wait..."}

from werkzeug.exceptions import RequestEntityTooLarge

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    flash("File is too large. Maximum upload size is 30 MB.")
    return redirect(request.url)

@app.route('/cancel_embedding', methods=['POST'])
def cancel_embedding():
    global embedding_status
    if embedding_status["running"]:
        embedding_status["running"] = False
        embedding_status["message"] = "Embedding cancelled."
        # Optionally, delete uploaded files.
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        return {"status": "cancelled", "message": "Embedding cancelled and uploads removed."}
    else:
        return {"status": "not_running", "message": "No embedding process is running."}

@app.route('/embedding_status', methods=['GET'])
def embedding_status_route():
    return embedding_status

@app.route('/qa', methods=['GET', 'POST'])
def qa():
    global vectorstore
    answer = None
    question = None
    sources = None
    embedded_docs = None

    # Attempt to load all embedded documents from the FAISS index.
    try:
        all_docs = []
        if vectorstore is not None:
            if hasattr(vectorstore.docstore, "docs"):
                all_docs = list(vectorstore.docstore.docs.values())
            else:
                # Fallback: iterate over vectorstore.docstore.__dict__ and filter for document-like objects.
                for value in vectorstore.docstore.__dict__.values():
                    if isinstance(value, dict):
                        for doc in value.values():
                            if hasattr(doc, "page_content"):
                                all_docs.append(doc)
            embedded_docs = list({doc.metadata.get("source", "unknown") for doc in all_docs})
        else:
            embedded_docs = []
            flash("Vectorstore is not loaded. Please run the embedding process first.")
    except Exception as e:
        embedded_docs = []
        flash(f"Error loading embedded documents: {e}")

    if request.method == "POST":
        question = request.form.get("question")
        if question:
            try:
                answer, sources = answer_question(question)
            except Exception as e:
                flash(f"Error generating answer: {e}")
        else:
            flash("Please enter a question.")
    return render_template("qa.html", question=question, answer=answer, sources=sources, embedded_docs=embedded_docs)

if __name__ == '__main__':
    app.run(debug=True)
