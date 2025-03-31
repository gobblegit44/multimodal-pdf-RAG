import os
import ollama
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import io
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
import hashlib
import json
from typing import List, Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiDocRAG:
    def __init__(self, embedding_model: str = 'nomic-embed-text', 
                 language_model: str = 'llama2',
                 vector_db_path: str = 'vector_db.json'):
        """
        Initialize the Multi-Document RAG system.
        
        Args:
            embedding_model: Name of the Ollama model to use for embeddings
            language_model: Name of the Ollama model to use for chat
            vector_db_path: Path to save/load the vector database
        """
        self.embedding_model = embedding_model
        self.language_model = language_model
        self.vector_db_path = vector_db_path
        self.vector_db = self._load_vector_db()
        
    def _load_vector_db(self) -> List[Dict[str, Any]]:
        """Load the vector database from file if it exists."""
        if os.path.exists(self.vector_db_path):
            try:
                with open(self.vector_db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading vector database: {e}")
                return []
        return []
    
    def _save_vector_db(self):
        """Save the vector database to file."""
        try:
            with open(self.vector_db_path, 'w') as f:
                json.dump(self.vector_db, f)
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for a file to track changes."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file, including both regular text and text from images."""
        extracted_text = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract regular text
                text = page.get_text()
                if text.strip():
                    extracted_text.append(text)
                
                # Extract text from images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert image bytes to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Extract text from image using pytesseract
                    try:
                        image_text = pytesseract.image_to_string(image)
                        if image_text.strip():
                            extracted_text.append(image_text)
                    except Exception as e:
                        logger.warning(f"Error extracting text from image on page {page_num + 1}: {str(e)}")
            
            doc.close()
            return "\n".join(extracted_text)
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return ""
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image file."""
        try:
            image = Image.open(image_path)
            return pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return ""
    
    def add_document(self, file_path: str) -> bool:
        """
        Add a document to the RAG system.
        
        Args:
            file_path: Path to the document (PDF or image)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
                
            # Check if file is already processed
            file_hash = self._get_file_hash(str(file_path))
            if any(doc['file_hash'] == file_hash for doc in self.vector_db):
                logger.info(f"File already processed: {file_path}")
                return True
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text = self._extract_text_from_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                text = self._extract_text_from_image(str(file_path))
            else:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return False
            
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return False
            
            # Split text into chunks
            chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            
            # Process chunks in batches
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                for chunk in batch:
                    try:
                        embedding = ollama.embed(model=self.embedding_model, input=chunk)['embeddings'][0]
                        self.vector_db.append({
                            'text': chunk,
                            'embedding': embedding,
                            'file_path': str(file_path),
                            'file_hash': file_hash,
                            'chunk_index': i + len(self.vector_db)
                        })
                    except Exception as e:
                        logger.error(f"Error getting embedding for chunk: {str(e)}")
                
                # Save after each batch
                self._save_vector_db()
                logger.info(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks from {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {str(e)}")
            return False
    
    def add_directory(self, directory_path: str) -> Dict[str, bool]:
        """
        Add all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Dict[str, bool]: Dictionary mapping file paths to success status
        """
        results = {}
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return results
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                results[str(file_path)] = self.add_document(str(file_path))
        
        return results
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum([x * y for x, y in zip(a, b)])
        norm_a = sum([x ** 2 for x in a]) ** 0.5
        norm_b = sum([x ** 2 for x in b]) ** 0.5
        return dot_product / (norm_a * norm_b)
    
    def retrieve(self, query: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks based on the query.
        
        Args:
            query: The search query
            top_n: Number of results to return
            
        Returns:
            List of dictionaries containing chunks and their metadata
        """
        try:
            query_embedding = ollama.embed(model=self.embedding_model, input=query)['embeddings'][0]
            similarities = []
            
            for doc in self.vector_db:
                similarity = self.cosine_similarity(query_embedding, doc['embedding'])
                similarities.append((doc, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_n]
        except Exception as e:
            logger.error(f"Error retrieving results: {str(e)}")
            return []
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and generate a response.
        
        Args:
            query: The user's question
            
        Returns:
            str: The generated response
        """
        try:
            # First, identify distinct topics and generate search queries
            topic_identification_prompt = '''You are a helpful assistant. Based on the user's question, identify the distinct topics/aspects that need to be addressed and generate a separate search query for each one. Format your response as a list of search queries, one per line. Only output the search queries, nothing else.'''
            
            query_response = ollama.chat(
                model=self.language_model,
                messages=[
                    {'role': 'system', 'content': topic_identification_prompt},
                    {'role': 'user', 'content': query},
                ]
            )
            
            search_queries = query_response['message']['content'].strip().split('\n')
            logger.info('\nGenerated search queries:')
            for q in search_queries:
                logger.info(f' - {q}')
            
            # Retrieve knowledge for each search query
            all_retrieved_knowledge = []
            for q in search_queries:
                retrieved = self.retrieve(q)
                all_retrieved_knowledge.extend(retrieved)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_retrieved_knowledge = []
            for doc, similarity in all_retrieved_knowledge:
                if doc['text'] not in seen:
                    seen.add(doc['text'])
                    unique_retrieved_knowledge.append((doc, similarity))
            
            logger.info('\nRetrieved knowledge:')
            for doc, similarity in unique_retrieved_knowledge:
                logger.info(f' - (similarity: {similarity:.2f}) From {doc["file_path"]}: {doc["text"]}')
            
            # Generate response
            instruction_prompt = f'''You are a helpful chatbot.
            Use only the following pieces of context to answer the question. Make sure to address all aspects of the question. Don't make up any new information.
            For each piece of information, cite the source file in parentheses.
            
            Context:
            {'\n'.join([f' - From {doc["file_path"]}: {doc["text"]}' for doc, _ in unique_retrieved_knowledge])}
            '''
            
            stream = ollama.chat(
                model=self.language_model,
                messages=[
                    {'role': 'system', 'content': instruction_prompt},
                    {'role': 'user', 'content': query},
                ],
                stream=True,
            )
            
            response = []
            print('\nChatbot response:')
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                response.append(content)
            
            return ''.join(response)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return "I apologize, but I encountered an error while processing your query. Please try again."

def main():
    # Initialize the RAG system
    rag = MultiDocRAG()
    
    while True:
        print("\nMulti-Document RAG System")
        print("1. Add a document")
        print("2. Add a directory")
        print("3. Ask a question")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            file_path = input("Enter the path to the document: ")
            if rag.add_document(file_path):
                print("Document added successfully!")
            else:
                print("Failed to add document.")
        
        elif choice == '2':
            dir_path = input("Enter the path to the directory: ")
            results = rag.add_directory(dir_path)
            success_count = sum(1 for success in results.values() if success)
            print(f"\nProcessed {success_count}/{len(results)} documents successfully.")
        
        elif choice == '3':
            query = input("\nEnter your question: ")
            response = rag.process_query(query)
            print("\nResponse:", response)
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 