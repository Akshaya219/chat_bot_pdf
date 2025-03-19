import os
import tempfile
import pickle
import sqlite3
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Configure API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Set GOOGLE_API_KEY in Streamlit Cloud secrets.")
    st.stop()
genai.configure(api_key=api_key)

# Initialize Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Database connection
DB_PATH = "vector_store.db"

def create_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS pdf_embeddings (pdf_name TEXT PRIMARY KEY, faiss_index BLOB)")
    conn.commit()
    conn.close()

def save_faiss_index(pdf_name, faiss_index):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    faiss_bytes = pickle.dumps(faiss_index)
    c.execute("INSERT OR REPLACE INTO pdf_embeddings (pdf_name, faiss_index) VALUES (?, ?)", (pdf_name, faiss_bytes))
    conn.commit()
    conn.close()

def load_faiss_index(pdf_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT faiss_index FROM pdf_embeddings WHERE pdf_name = ?", (pdf_name,))
    row = c.fetchone()
    conn.close()
    if row:
        return pickle.loads(row[0])
    return None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_per_page = [(page_num, page.get_text("text")) for page_num, page in enumerate(doc)]
    doc.close()
    return text_per_page

# Function to index PDF text with FAISS
def index_pdf_text(text_per_page):
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for page_num, text in text_per_page:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={'page': page_num})
            documents.append(doc)
    vector_store = FAISS.from_documents(documents, embedding_function)
    return vector_store

# Function to query Gemini API with context
def query_gemini(prompt, context):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Context: {context}\nUser Query: {prompt}\nProvide a concise answer.")
        return response.text
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}"

# Function to search PDF and answer questions
def search_pdf_and_answer(query, vector_store):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    answer = query_gemini(query, context)
    return answer

# Streamlit UI
st.title("📄 PDF Chatbot with FAISS Database")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    pdf_name = uploaded_file.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    
    create_db()  # Ensure DB exists

    # Load existing FAISS index from DB if available
    vector_store = load_faiss_index(pdf_name)

    if not vector_store:
        with st.spinner("Processing PDF... Please wait..."):
            text_per_page = extract_text_from_pdf(temp_path)
            vector_store = index_pdf_text(text_per_page)
            save_faiss_index(pdf_name, vector_store)
        st.success("PDF successfully indexed and saved! ✅")
    else:
        st.success("Loaded existing index from the database! ✅")

    query = st.text_input("Ask a question from the PDF:")

    if query:
        with st.spinner("Generating response..."):
            answer = search_pdf_and_answer(query, vector_store)
        st.write("### 🤖 Answer:")
        st.write(answer)
