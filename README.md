# PDF Chatbot with Database Integration

A PDF chatbot that uses Google's Gemini API to answer questions about PDF documents, with database integration for caching and improved performance.

## Features

- Upload and process PDF documents
- Extract text and images from PDFs
- Create vector embeddings for semantic search
- Cache processed PDFs to avoid re-processing
- Store and retrieve query history
- Fast responses with database caching
- Display relevant images alongside answers

## Requirements

- Python 3.8+
- Google API key for Gemini
- Dependencies listed in `requirements.txt`

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example` and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Docker Deployment

To run the application in a Docker container:

1. Build the Docker image:
   ```
   docker build -t pdf-chatbot .
   ```
2. Run the container:
   ```
   docker run -p 8501:8501 pdf-chatbot
   ```
3. Access the application at http://localhost:8501

## Database Structure

The application uses SQLite for data storage with the following schema:

### PDFs Table
- `id`: Unique identifier for the PDF
- `filename`: Original filename
- `hash`: MD5 hash for duplicate detection
- `upload_date`: When the PDF was uploaded
- `vector_store_path`: Path to stored vector embeddings

### Queries Table
- `id`: Unique identifier for the query
- `pdf_id`: Reference to the PDF
- `query`: User's question
- `response`: AI response
- `timestamp`: When the query was made

## Project Structure

- `app.py` - Main application file
- `requirements.txt` - Python dependencies
- `pdf_chatbot.db` - SQLite database file (created automatically)
- `vector_stores/` - Directory for storing vector embeddings
- `image_cache/` - Directory for storing extracted images

## Performance Improvements

1. **Caching**: The application caches PDF processing results and query responses
2. **Vector Indexing**: Uses FAISS for efficient similarity search
3. **Database Storage**: Stores processing results and query history
4. **Image Extraction**: Extracts and caches images for faster retrieval

## Usage Tips

1. For best results, use PDFs with clear text
2. Large PDFs may take longer to process initially but will be cached for subsequent use
3. Use specific questions for more accurate answers
