# app.py
import os
import tempfile
import sqlite3
import uuid
import hashlib
from datetime import datetime
import shutil
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Constants
DATA_DIR = "./data"
DEFAULT_PDF = "default.pdf"
DEFAULT_VECTOR_STORE = os.path.join(DATA_DIR, "vector_store.faiss")
DB_PATH = os.path.join(DATA_DIR, "pdf_chatbot.db")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Database initialization
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.executescript('''
            CREATE TABLE IF NOT EXISTS pdfs (
                id TEXT PRIMARY KEY,
                filename TEXT,
                hash TEXT,
                upload_date TIMESTAMP,
                vector_store_path TEXT
            );
            CREATE TABLE IF NOT EXISTS queries (
                id TEXT PRIMARY KEY,
                pdf_id TEXT,
                query TEXT,
                response TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
            );
        ''')
        conn.commit()
        return conn
    except sqlite3.Error as e:
        st.error(f"Database error: {e}. Using in-memory fallback.")
        return sqlite3.connect(":memory:")

# Configure API
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Set GOOGLE_API_KEY in environment or Streamlit secrets.")
    st.stop()
genai.configure(api_key=api_key)

# Cached model loading
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Compute file hash
def compute_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Extract text from PDF (cached)
@st.cache_data
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_per_page = [(page_num, page.get_text("text")) for page_num, page in enumerate(doc)]
    doc.close()
    return text_per_page

# Extract images on demand
def extract_images_from_pages(pdf_path, page_nums):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in page_nums:
        page = doc[page_num]
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=DATA_DIR) as tmp:
                tmp.write(base_image["image"])
                images.append(tmp.name)
    doc.close()
    return images

# Index PDF text
def index_pdf_text(text_per_page, pdf_id):
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for page_num, text in text_per_page:
        chunks = text_splitter.split_text(text)
        documents.extend(Document(page_content=chunk, metadata={'page': page_num, 'pdf_id': pdf_id}) for chunk in chunks)
    
    vector_store_path = os.path.join(DATA_DIR, f"vector_store_{pdf_id}")
    vector_store = FAISS.from_documents(documents, get_embedding_function())
    vector_store.save_local(vector_store_path)
    return vector_store, vector_store_path

# Load vector store
def load_vector_store(path):
    return FAISS.load_local(path, get_embedding_function())

# Query Gemini API
@st.cache_data(ttl=3600)
def query_gemini(prompt, context):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        f"Context: {context}\nUser Query: {prompt}\nShort, concise answer for exam prep."
    )
    return response.text

# Search PDF and answer
def search_pdf_and_answer(query, vector_store, pdf_path, pdf_id, conn):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join(doc.page_content for doc in docs)
    
    cursor = conn.cursor()
    cursor.execute("SELECT response FROM queries WHERE pdf_id = ? AND query = ?", (pdf_id, query))
    cached = cursor.fetchone()
    if cached:
        answer = cached[0]
    else:
        answer = query_gemini(query, context)
        cursor.execute("INSERT INTO queries VALUES (?, ?, ?, ?, ?)", 
                       (str(uuid.uuid4()), pdf_id, query, answer, datetime.now()))
        conn.commit()
    
    page_nums = set(doc.metadata['page'] for doc in docs)
    images = extract_images_from_pages(pdf_path, page_nums)
    return answer, images

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
    conn = init_db()
    
    st.title("📄 Smart PDF Chatbot")
    tab1, tab2 = st.tabs(["Chat", "History"])

    # Load default vector store if available
    pdf_path = DEFAULT_PDF if os.path.exists(DEFAULT_PDF) else None
    vector_store = load_vector_store(DEFAULT_VECTOR_STORE) if os.path.exists(DEFAULT_VECTOR_STORE) else None
    pdf_id = "default"

    with tab1:
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=DATA_DIR) as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name
            
            file_hash = compute_file_hash(pdf_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, vector_store_path FROM pdfs WHERE hash = ?", (file_hash,))
            existing = cursor.fetchone()
            
            if existing and os.path.exists(existing[1]):
                pdf_id, vector_store_path = existing
                vector_store = load_vector_store(vector_store_path)
                st.success("Loaded cached PDF!")
            else:
                with st.spinner("Processing PDF..."):
                    pdf_id = str(uuid.uuid4())
                    text_per_page = extract_text_from_pdf(pdf_path)
                    vector_store, vector_store_path = index_pdf_text(text_per_page, pdf_id)
                    cursor.execute("INSERT INTO pdfs VALUES (?, ?, ?, ?, ?)", 
                                   (pdf_id, uploaded_file.name, file_hash, datetime.now(), vector_store_path))
                    conn.commit()
                st.success("PDF indexed!")

        if vector_store and pdf_path:
            query = st.text_input("Ask a question:")
            if query:
                with st.spinner("Thinking..."):
                    answer, images = search_pdf_and_answer(query, vector_store, pdf_path, pdf_id, conn)
                st.markdown(f"### Answer:\n{answer}")
                if images:
                    st.write("#### Images:")
                    cols = st.columns(min(3, len(images)))
                    for i, img in enumerate(images):
                        cols[i % 3].image(img, use_column_width=True)

    with tab2:
        st.subheader("Query History")
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename FROM pdfs")
        pdfs = cursor.fetchall()
        if pdfs:
            selected = st.selectbox("Select PDF", [f"{p[1]} ({p[0]})" for p in pdfs], format_func=lambda x: x.split(" (")[0])
            pdf_id = selected.split(" (")[1].rstrip(")")
            cursor.execute("SELECT query, response, timestamp FROM queries WHERE pdf_id = ? ORDER BY timestamp DESC", (pdf_id,))
            for q, r, t in cursor.fetchall():
                with st.expander(f"Q: {q}"):
                    st.markdown(f"**Question:** {q}\n**Answer:** {r}\n**Time:** {t}")

if __name__ == "__main__":
    main()
