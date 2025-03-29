# Multi-modal PDF RAG system
 
The system uses:
## PyMuPDF (fitz) for PDF processing
## Tesseract OCR for image text extraction
## Ollama for embeddings and language model
## Cosine similarity for semantic search

The solution includes:

A Python script (multimodal_pdf_rag.py) that
- Extracts text from both regular PDF content and images using OCR
- Saves the extracted text to a file
- Implements a RAG (Retrieval-Augmented Generation) system for context-aware responses
- Provides an interactive interface for asking questions about the PDF content

A requirements.txt file with all necessary dependencies

Install the required dependencies: pip install -r requirements.txt

Make sure Ollama is running on your system
Pull the required models by running these commands in your terminal:

ollama pull nomic-embed-text
ollama pull llama2

Install Tesseract OCR on your system:
For macOS: brew install tesseract
For Ubuntu/Debian: sudo apt-get install tesseract-ocr
For Windows: Download and install from the official GitHub releases
Run the script: python multimodal_pdf_rag.py

The system will:
1. Ask you to provide the path to your PDF file
2. Extract text from both regular content and images in the PDF
3. Save the extracted text to extracted_text.txt
4. Set up the RAG system with the extracted content
5. Allow you to ask questions about the PDF content interactively

Key features of the implementation:
- Multi-modal text extraction (both regular text and images)
- Efficient text chunking and embedding
- Context-aware question answering using RAG
- Interactive query interface
- Progress tracking during processing
- Error handling for image processing

