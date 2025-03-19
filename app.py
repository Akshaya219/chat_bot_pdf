# app.py
# Standard library imports
import os
import tempfile
import sqlite3
import uuid
import hashlib
from datetime import datetime
import shutil

# Third-party imports
import streamlit as st
try:
    import fitz  # PyMuPDF
except ImportError:
    st.error("PyMuPDF not installed correctly. Please check requirements.txt.")
    st.stop()

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Database initialization
def init_db():
    conn = sqlite3.connect('pdf_chatbot.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pdfs (
        id TEXT PRIMARY KEY,
        filename TEXT,
        hash TEXT,
        upload_date TIMESTAMP,
        vector_store_path TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS queries (
        id TEXT PRIMARY KEY,
        pdf_id TEXT,
        query TEXT,
        response TEXT,
        timestamp TIMESTAMP,
        FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
    )
    ''')
    conn.commit()
    return conn

# Configure API with environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Set GOOGLE_API_KEY in Streamlit Cloud secrets.")
    st.stop()
genai.configure(api_key=api_key)

# Initialize models
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to compute file hash
def compute_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

# Function to extract text and images from PDF
def extract_text_and_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_per_page = []
    images_per_page = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text_per_page.append((page_num, text))
        images = page.get_images(full=True)
        images_per_page[page_num] = []
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # Create a persistent image directory
            os.makedirs("image_cache", exist_ok=True)
            image_path = f"image_cache/{uuid.uuid4()}.png"
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
                images_per_page[page_num].append(image_path)
    doc.close()
    return text_per_page, images_per_page

# Function to index PDF text with page metadata
def index_pdf_text(text_per_page, pdf_id):
    documents = []
    for page_num, text in text_per_page:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={'page': page_num, 'pdf_id': pdf_id})
            documents.append(doc)
    
    # Create directory for vector stores
    os.makedirs("vector_stores", exist_ok=True)
    vector_store_path = f"vector_stores/{pdf_id}"
    
    embedding_function = get_embedding_function()
    vector_store = FAISS.from_documents(documents, embedding_function)
    vector_store.save_local(vector_store_path)
    
    return vector_store, vector_store_path

# Function to load existing vector store
def load_vector_store(vector_store_path):
    embedding_function = get_embedding_function()
    return FAISS.load_local(vector_store_path, embedding_function)

# Function to query Gemini API with concise prompt
@st.cache_data(ttl=3600)  # Cache responses for an hour
def query_gemini(prompt, context):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"Context: {context}\nUser Query: {prompt}\nProvide a short and concise answer suitable for exam preparation."
        )
        return response.text
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}"

# Function to search PDF and answer with images
def search_pdf_and_answer(query, vector_store, images_per_page, pdf_id, conn):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Check if this query has been asked before
    cursor = conn.cursor()
    cursor.execute("SELECT response FROM queries WHERE pdf_id = ? AND query = ?", (pdf_id, query))
    existing_response = cursor.fetchone()
    
    if existing_response:
        answer = existing_response[0]
    else:
        answer = query_gemini(query, context)
        # Store the query and response
        query_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO queries VALUES (?, ?, ?, ?, ?)",
            (query_id, pdf_id, query, answer, datetime.now())
        )
        conn.commit()
    
    page_nums = set(doc.metadata['page'] for doc in docs)
    relevant_images = []
    for page_num in page_nums:
        if page_num in images_per_page:
            relevant_images.extend(images_per_page.get(page_num, []))
    
    return answer, relevant_images

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
    
    # Initialize database
    conn = init_db()
    
    st.title("📄 Smart PDF Chatbot with Gemini API 🤖")
    st.sidebar.header("Options")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Chat with PDF", "History", "Settings"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            
            file_hash = compute_file_hash(temp_path)
            
            # Check if PDF already exists in database
            cursor = conn.cursor()
            cursor.execute("SELECT id, vector_store_path FROM pdfs WHERE hash = ?", (file_hash,))
            existing_pdf = cursor.fetchone()
            
            if existing_pdf:
                pdf_id, vector_store_path = existing_pdf
                st.success("This PDF has been processed before! Loading from cache... ✅")
                vector_store = load_vector_store(vector_store_path)
                
                # Load images
                cursor.execute("SELECT filename FROM pdfs WHERE id = ?", (pdf_id,))
                filename = cursor.fetchone()[0]
                
                # Extract images again as they might not be stored long-term
                text_per_page, images_per_page = extract_text_and_images_from_pdf(temp_path)
            else:
                with st.spinner("Processing PDF... Please wait..."):
                    pdf_id = str(uuid.uuid4())
                    text_per_page, images_per_page = extract_text_and_images_from_pdf(temp_path)
                    vector_store, vector_store_path = index_pdf_text(text_per_page, pdf_id)
                    
                    # Store PDF information in database
                    cursor.execute(
                        "INSERT INTO pdfs VALUES (?, ?, ?, ?, ?)",
                        (pdf_id, uploaded_file.name, file_hash, datetime.now(), vector_store_path)
                    )
                    conn.commit()
                st.success("PDF successfully indexed! ✅")
            
            # Chat interface
            st.subheader("Ask questions about your PDF")
            query = st.text_input("Your question:")
            
            if query:
                with st.spinner("Thinking..."):
                    answer, relevant_images = search_pdf_and_answer(query, vector_store, images_per_page, pdf_id, conn)
                
                st.write("### 🤖 Answer:")
                st.markdown(answer)
                
                if relevant_images:
                    st.write("#### Relevant Images from PDF:")
                    cols = st.columns(min(3, len(relevant_images)))
                    for i, img_path in enumerate(relevant_images):
                        try:
                            cols[i % 3].image(img_path, use_column_width=True)
                        except Exception as e:
                            cols[i % 3].error(f"Could not load image: {e}")
    
    with tab2:
        st.subheader("Query History")
        
        # Get list of PDFs
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename FROM pdfs ORDER BY upload_date DESC")
        pdfs = cursor.fetchall()
        
        if pdfs:
            selected_pdf = st.selectbox(
                "Select PDF to view history", 
                options=[f"{pdf[1]} ({pdf[0]})" for pdf in pdfs],
                format_func=lambda x: x.split(" (")[0]
            )
            
            if selected_pdf:
                pdf_id = selected_pdf.split(" (")[1].rstrip(")")
                
                # Retrieve query history
                cursor.execute(
                    "SELECT query, response, timestamp FROM queries WHERE pdf_id = ? ORDER BY timestamp DESC", 
                    (pdf_id,)
                )
                queries = cursor.fetchall()
                
                if queries:
                    for i, (query, response, timestamp) in enumerate(queries):
                        with st.expander(f"Q: {query} ({timestamp})"):
                            st.markdown(f"**Question:** {query}")
                            st.markdown(f"**Answer:** {response}")
                else:
                    st.info("No query history for this PDF.")
        else:
            st.info("No PDFs uploaded yet.")
    
    with tab3:
        st.subheader("Settings")
        if st.button("Clear all data"):
            if st.checkbox("I understand this will delete all PDFs and query history"):
                # Delete database
                conn.close()
                if os.path.exists("pdf_chatbot.db"):
                    os.remove("pdf_chatbot.db")
                
                # Delete vector stores
                if os.path.exists("vector_stores"):
                    shutil.rmtree("vector_stores")
                
                # Delete image cache
                if os.path.exists("image_cache"):
                    shutil.rmtree("image_cache")
                
                st.success("All data cleared successfully!")
                st.experimental_rerun()

if __name__ == "__main__":
    main()
