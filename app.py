import os
import tempfile
import sqlite3
import uuid
import hashlib
from datetime import datetime
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
DB_PATH = os.path.join(DATA_DIR, "pdf_chatbot.db")
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
        st.error(f"Database error: {e}")
        return sqlite3.connect(":memory:")

# Configure API
def configure_api():
    api_key = os.getenv("GOOGLE_API_KEY") 
    if not api_key:
        api_key = st.text_input("Enter Google API Key:", type="password")
        if not api_key:
            st.warning("Please provide a Google API Key to continue.")
            st.stop()
    genai.configure(api_key=api_key)
    return api_key

# Helper functions
@st.cache_resource
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def compute_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

@st.cache_data
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_per_page = [(page_num, page.get_text("text")) for page_num, page in enumerate(doc)]
    doc.close()
    return text_per_page

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

# Load vector store safely
def load_vector_store(path):
    try:
        return FAISS.load_local(path, get_embedding_function(), allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

# Query Gemini API with content verification
def query_gemini(prompt, context, detail_level="medium"):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    system_prompt = {
        "short": "Provide a very concise answer (1-2 sentences) based ONLY on the context provided. If the answer isn't in the context, say 'This information is not in the document.'",
        "medium": "Provide a balanced answer (3-5 sentences) based ONLY on the context provided. If the answer isn't in the context, say 'This information is not in the document.'",
        "detailed": "Provide a comprehensive answer with specific details from the context. If the answer isn't in the context, say 'This information is not in the document.'"
    }
    
    try:
        response = model.generate_content(
            f"Context from PDF: {context}\n\nUser Query: {prompt}\n\n{system_prompt[detail_level]}"
        )
        return response.text
    except Exception as e:
        return f"Error querying AI: {e}"

# Search PDF and answer
def search_pdf_and_answer(query, vector_store, pdf_path, pdf_id, conn, detail_level):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join(doc.page_content for doc in docs)
    
    answer = query_gemini(query, context, detail_level)
    
    # Save query to history
    cursor = conn.cursor()
    cursor.execute("INSERT INTO queries VALUES (?, ?, ?, ?, ?)", 
                   (str(uuid.uuid4()), pdf_id, query, answer, datetime.now()))
    conn.commit()
    
    page_nums = set(doc.metadata['page'] for doc in docs)
    images = extract_images_from_pages(pdf_path, page_nums)
    return answer, images, [doc.page_content for doc in docs]

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
    
    st.title("📄 Smart PDF Chatbot")
    
    # Initialize session state
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pdf_id" not in st.session_state:
        st.session_state.pdf_id = None
        
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        detail_level = st.radio(
            "Answer Detail Level:",
            ["short", "medium", "detailed"],
            index=1
        )
        
        # Configure API Key
        api_key = configure_api()
    
    # Main layout with tabs
    tab1, tab2 = st.tabs(["Upload & Chat", "History"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("1. Upload your PDF")
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            
            if uploaded_file:
                with st.spinner("Processing PDF..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=DATA_DIR) as tmp:
                        tmp.write(uploaded_file.read())
                        pdf_path = tmp.name
                    
                    file_hash = compute_file_hash(pdf_path)
                    conn = init_db()
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, vector_store_path FROM pdfs WHERE hash = ?", (file_hash,))
                    existing = cursor.fetchone()
                    
                    if existing and os.path.exists(existing[1]):
                        st.session_state.pdf_id, vector_store_path = existing
                        st.session_state.vector_store = load_vector_store(vector_store_path)
                        st.session_state.pdf_path = pdf_path
                        st.success("PDF loaded from cache! Ready to chat.")
                    else:
                        st.session_state.pdf_id = str(uuid.uuid4())
                        text_per_page = extract_text_from_pdf(pdf_path)
                        st.session_state.vector_store, vector_store_path = index_pdf_text(text_per_page, st.session_state.pdf_id)
                        st.session_state.pdf_path = pdf_path
                        cursor.execute("INSERT INTO pdfs VALUES (?, ?, ?, ?, ?)", 
                                    (st.session_state.pdf_id, uploaded_file.name, file_hash, datetime.now(), vector_store_path))
                        conn.commit()
                        st.success("PDF indexed successfully! Ready to chat.")
        
        with col2:
            st.subheader("2. Ask questions about your PDF")
            if st.session_state.vector_store and st.session_state.pdf_path:
                query = st.text_input("Your question:")
                
                conn = init_db()
                if query:
                    with st.spinner("Searching..."):
                        answer, images, source_texts = search_pdf_and_answer(
                            query, 
                            st.session_state.vector_store, 
                            st.session_state.pdf_path, 
                            st.session_state.pdf_id, 
                            conn,
                            detail_level
                        )
                    
                    st.markdown(f"### Answer")
                    st.write(answer)
                    
                    with st.expander("View source context"):
                        for i, text in enumerate(source_texts):
                            st.markdown(f"**Source {i+1}:**")
                            st.write(text)
                    
                    if images:
                        with st.expander("View related images from PDF"):
                            cols = st.columns(min(3, len(images)))
                            for i, img in enumerate(images):
                                cols[i % 3].image(img, use_column_width=True)
            else:
                st.info("Please upload a PDF first to start chatting")
    
    # History tab
    with tab2:
        st.subheader("Query History")
        conn = init_db()
        if st.session_state.pdf_id:
            cursor = conn.cursor()
            cursor.execute("SELECT query, response, timestamp FROM queries WHERE pdf_id = ? ORDER BY timestamp DESC LIMIT 10", 
                          (st.session_state.pdf_id,))
            history = cursor.fetchall()
            
            if history:
                for q, r, t in history:
                    with st.expander(f"Q: {q}"):
                        st.write(f"**Answer:** {r}")
                        st.write(f"**Time:** {t}")
            else:
                st.info("No query history yet. Start chatting with your PDF!")
        else:
            st.info("Upload a PDF to see your query history.")

if __name__ == "__main__":
    main()
