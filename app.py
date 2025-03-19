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

# Get the database path - ensuring it's accessible in the Streamlit environment
def get_db_path():
    # Try to use a location that's definitely writable in Streamlit Cloud
    if os.path.exists("/mount/data"):
        return "/mount/data/pdf_chatbot.db"
    elif os.path.exists("/tmp"):
        return "/tmp/pdf_chatbot.db"
    else:
        # Use current directory as last resort
        return "pdf_chatbot.db"

# Database initialization
def init_db():
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
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
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        # Fall back to in-memory database if file access fails
        st.warning("Falling back to in-memory database. Your data won't be persisted between sessions.")
        conn = sqlite3.connect(":memory:")
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

# Get paths for storage
def get_storage_paths():
    # Try to use locations that are definitely writable in Streamlit Cloud
    if os.path.exists("/mount/data"):
        base_path = "/mount/data"
    elif os.path.exists("/tmp"):
        base_path = "/tmp"
    else:
        base_path = "."
        
    vector_store_path = os.path.join(base_path, "vector_stores")
    image_cache_path = os.path.join(base_path, "image_cache")
    
    # Create directories if they don't exist
    os.makedirs(vector_store_path, exist_ok=True)
    os.makedirs(image_cache_path, exist_ok=True)
    
    return vector_store_path, image_cache_path

# Configure API with environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Try to get from Streamlit secrets
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
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
    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    except Exception as e:
        st.warning(f"Error computing file hash: {e}")
        return str(uuid.uuid4())  # Fallback to random UUID if hash fails

# Function to extract text and images from PDF
def extract_text_and_images_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text_per_page = []
        images_per_page = {}
        
        _, image_cache_path = get_storage_paths()
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            text_per_page.append((page_num, text))
            images = page.get_images(full=True)
            images_per_page[page_num] = []
            for img in images:
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_path = os.path.join(image_cache_path, f"{uuid.uuid4()}.png")
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                        images_per_page[page_num].append(image_path)
                except Exception as e:
                    st.warning(f"Error extracting image {xref}: {e}")
        doc.close()
        return text_per_page, images_per_page
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return [], {}

# Function to index PDF text with page metadata
def index_pdf_text(text_per_page, pdf_id):
    documents = []
    for page_num, text in text_per_page:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={'page': page_num, 'pdf_id': pdf_id})
            documents.append(doc)
    
    vector_store_path, _ = get_storage_paths()
    pdf_vector_path = os.path.join(vector_store_path, pdf_id)
    
    embedding_function = get_embedding_function()
    vector_store = FAISS.from_documents(documents, embedding_function)
    
    try:
        vector_store.save_local(pdf_vector_path)
        return vector_store, pdf_vector_path
    except Exception as e:
        st.warning(f"Could not save vector store to disk: {e}")
        return vector_store, None  # Return None for path if saving fails

# Function to load existing vector store
def load_vector_store(vector_store_path):
    try:
        embedding_function = get_embedding_function()
        return FAISS.load_local(vector_store_path, embedding_function)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

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
    try:
        # Default response in case of errors
        default_response = "I couldn't find an answer to that question in the PDF."
        
        if vector_store is None:
            return default_response, []
        
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return default_response, []
        
        context = "\n".join([doc.page_content for doc in docs])
        
        # Try to get cached response
        try:
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
        except Exception as e:
            st.warning(f"Database error when saving query: {e}")
            answer = query_gemini(query, context)
        
        page_nums = set(doc.metadata['page'] for doc in docs)
        relevant_images = []
        for page_num in page_nums:
            if page_num in images_per_page:
                relevant_images.extend(images_per_page.get(page_num, []))
        
        return answer, relevant_images
    except Exception as e:
        st.error(f"Error searching PDF: {e}")
        return "I encountered an error while searching the PDF.", []

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
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id, vector_store_path FROM pdfs WHERE hash = ?", (file_hash,))
                existing_pdf = cursor.fetchone()
                
                if existing_pdf and existing_pdf[1] and os.path.exists(existing_pdf[1]):
                    pdf_id, vector_store_path = existing_pdf
                    st.success("This PDF has been processed before! Loading from cache... ✅")
                    vector_store = load_vector_store(vector_store_path)
                    
                    # Extract images again as they might not be stored long-term
                    text_per_page, images_per_page = extract_text_and_images_from_pdf(temp_path)
                else:
                    with st.spinner("Processing PDF... Please wait..."):
                        pdf_id = str(uuid.uuid4())
                        text_per_page, images_per_page = extract_text_and_images_from_pdf(temp_path)
                        vector_store, vector_store_path = index_pdf_text(text_per_page, pdf_id)
                        
                        # Store PDF information in database
                        try:
                            cursor.execute(
                                "INSERT INTO pdfs VALUES (?, ?, ?, ?, ?)",
                                (pdf_id, uploaded_file.name, file_hash, datetime.now(), vector_store_path)
                            )
                            conn.commit()
                        except Exception as e:
                            st.warning(f"Could not store PDF info in database: {e}")
                    st.success("PDF successfully indexed! ✅")
            except Exception as e:
                st.error(f"Database error: {e}")
                with st.spinner("Processing PDF... Please wait..."):
                    pdf_id = str(uuid.uuid4())
                    text_per_page, images_per_page = extract_text_and_images_from_pdf(temp_path)
                    vector_store, _ = index_pdf_text(text_per_page, pdf_id)
                st.success("PDF successfully indexed! ✅")
            
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
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
                            if os.path.exists(img_path):
                                cols[i % 3].image(img_path, use_column_width=True)
                        except Exception as e:
                            cols[i % 3].error(f"Could not load image: {e}")
    
    with tab2:
        st.subheader("Query History")
        
        try:
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
                            with st.expander(f"Q: {query}"):
                                st.markdown(f"**Question:** {query}")
                                st.markdown(f"**Answer:** {response}")
                                st.caption(f"Asked on: {timestamp}")
                    else:
                        st.info("No query history for this PDF.")
            else:
                st.info("No PDFs uploaded yet.")
        except Exception as e:
            st.error(f"Error accessing history: {e}")
    
    with tab3:
        st.subheader("Settings")
        st.write("#### Storage Locations")
        vector_store_path, image_cache_path = get_storage_paths()
        st.write(f"Vector store path: `{vector_store_path}`")
        st.write(f"Image cache path: `{image_cache_path}`")
        st.write(f"Database path: `{get_db_path()}`")
        
        st.write("#### Clear Data")
        if st.button("Clear all data"):
            confirm = st.checkbox("I understand this will delete all PDFs and query history")
            if confirm:
                try:
                    # Close connection
                    conn.close()
                    
                    # Delete database if it's a file
                    db_path = get_db_path()
                    if db_path != ":memory:" and os.path.exists(db_path):
                        os.remove(db_path)
                    
                    # Delete vector stores
                    vector_path, image_path = get_storage_paths()
                    if os.path.exists(vector_path):
                        shutil.rmtree(vector_path, ignore_errors=True)
                    
                    # Delete image cache
                    if os.path.exists(image_path):
                        shutil.rmtree(image_path, ignore_errors=True)
                    
                    st.success("All data cleared successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error clearing data: {e}")

if __name__ == "__main__":
    main()
