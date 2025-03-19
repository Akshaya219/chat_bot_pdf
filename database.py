import os
import psycopg2
from psycopg2.extras import execute_values
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.client = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
    
    def store_embeddings(self, documents):
        """Store documents with optimized batch processing"""
        self.client.add_documents(
            documents,
            ids=[f"{doc.metadata['pdf_hash']}_{doc.metadata['page']}" 
                 for doc in documents]
        )
    
    def get_store(self):
        return self.client

class PostgresManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432")
        )
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_metadata (
                    id SERIAL PRIMARY KEY,
                    pdf_hash VARCHAR(64) NOT NULL,
                    page_num INTEGER NOT NULL,
                    text_chunk TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS pdf_hash_idx 
                ON pdf_metadata (pdf_hash);
            """)
            self.conn.commit()

    def exists(self, pdf_hash):
        """Check if PDF exists in database"""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pdf_metadata WHERE pdf_hash = %s LIMIT 1",
                (pdf_hash,)
            )
            return cur.fetchone() is not None

    def store_metadata(self, pdf_hash, metadata):
        """Batch insert metadata"""
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """INSERT INTO pdf_metadata (pdf_hash, page_num, text_chunk)
                   VALUES %s""",
                [(pdf_hash, m['page'], m['text']) for m in metadata],
                page_size=100
            )
            self.conn.commit()
