# Standard library imports
import os
import tempfile
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from database import VectorDB, PostgresManager  # Local module

# Configure API with environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Set GOOGLE_API_KEY in Streamlit Cloud secrets.")
    st.stop()
genai.configure(api_key=api_key)

# Initialize components
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = VectorDB()
postgres_db = PostgresManager()

def get_pdf_hash(uploaded_file):
    """Generate content-based hash for PDF files"""
    return hashlib.sha256(uploaded_file.getbuffer()).hexdigest()

def process_page(page):
    """Parallel processing of PDF pages"""
    text = page.get_text("text")
    images = [img[0] for img in page.get_images(full=True)]
    return text, images

def extract_text_and_images(pdf_path):
    """Optimized PDF extraction with parallel processing"""
    doc = fitz.open(pdf_path)
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_page, doc))
    
    text_per_page = []
    images_per_page = {}
    for page_num, (text, images) in enumerate(results):
        text_per_page.append((page_num, text))
        images_per_page[page_num] = [doc.extract_image(xref)["image"] for xref in images]
    
    doc.close()
    return text_per_page, images_per_page

def index_documents(text_per_page, pdf_hash):
    """Optimized indexing with hybrid storage"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents = []
    metadata_list = []
    
    for page_num, text in text_per_page:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={'page': page_num, 'pdf_hash': pdf_hash}
            )
            documents.append(doc)
            metadata_list.append({
                'page': page_num,
                'text': chunk,
                'pdf_hash': pdf_hash
            })
    
    # Store in ChromaDB
    vector_db.store_embeddings(documents)
    
    # Store in PostgreSQL
    postgres_db.store_metadata(pdf_hash, metadata_list)
    
    return vector_db.get_store()

def query_llm(prompt, context):
    """Optimized LLM querying with caching"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        f"Context: {context}\nQuery: {prompt}\nProvide concise answer with page references:"
    )
    return response.text

# Streamlit UI
st.title("🚀 Smart PDF Analyzer")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    pdf_hash = get_pdf_hash(uploaded_file)
    
    if not postgres_db.exists(pdf_hash):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
        
        with st.spinner("🔄 Processing PDF..."):
            text_data, image_data = extract_text_and_images(tmp_file.name)
            vector_store = index_documents(text_data, pdf_hash)
        
        st.success("✅ PDF indexed successfully!")
    else:
        st.info("📚 Using cached version of this PDF")
        vector_store = vector_db.get_store()

    query = st.text_input("Ask about the document:")
    
    if query:
        with st.spinner("💡 Generating response..."):
            results = vector_store.similarity_search(query, k=4)
            context = "\n".join([f"Page {doc.metadata['page']+1}: {doc.page_content}" 
                               for doc in results])
            answer = query_llm(query, context)
            
            pages = {doc.metadata['page'] for doc in results}
            images = [image_data[page][0] for page in pages if page in image_data]
        
        st.markdown("### 📝 Answer")
        st.write(answer)
        
        if images:
            st.markdown("### 📷 Related Content")
            cols = st.columns(min(3, len(images)))
            for idx, img_bytes in enumerate(images[:3]):
                cols[idx % 3].image(img_bytes, use_column_width=True)
