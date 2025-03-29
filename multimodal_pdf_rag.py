import os
import ollama
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import io
import fitz  # PyMuPDF
import numpy as np

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file, including both regular text and text from images.
    """
    extracted_text = []
    
    # Open PDF with PyMuPDF
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
                print(f"Error extracting text from image on page {page_num + 1}: {str(e)}")
    
    doc.close()
    return "\n".join(extracted_text)

def save_text_to_file(text, output_file):
    """
    Save extracted text to a file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Text saved to {output_file}")

def setup_rag_system(text_chunks):
    """
    Set up the RAG system with the extracted text.
    """
    # Using Ollama's nomic-embed-text model for embeddings and llama2 for chat
    EMBEDDING_MODEL = 'nomic-embed-text'
    LANGUAGE_MODEL = 'llama2'
    
    VECTOR_DB = []
    
    def add_chunk_to_database(chunk):
        try:
            embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            VECTOR_DB.append((chunk, embedding))
        except Exception as e:
            print(f"Error getting embedding for chunk: {str(e)}")
            print("Make sure Ollama is running and the model is pulled. Try running: ollama pull nomic-embed-text")
    
    # Split text into chunks and add to database
    # Process chunks in batches for better organization
    batch_size = 10
    total_batches = (len(text_chunks) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(text_chunks))
        current_batch = text_chunks[start_idx:end_idx]
        
        print(f'\nProcessing batch {batch_num + 1}/{total_batches}:')
        for i, chunk in enumerate(current_batch):
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            # Add meaningful chunks to database
            add_chunk_to_database(chunk)
            print(f'  Added chunk {start_idx + i + 1}/{len(text_chunks)} to the database')
    
    return VECTOR_DB, EMBEDDING_MODEL, LANGUAGE_MODEL

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    """
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, VECTOR_DB, EMBEDDING_MODEL, top_n=3):
    """
    Retrieve relevant chunks based on the query.
    """
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def process_query(query, VECTOR_DB, EMBEDDING_MODEL, LANGUAGE_MODEL):
    """
    Process a user query and generate a response.
    """
    try:
        # First, identify distinct topics and generate search queries
        topic_identification_prompt = '''You are a helpful assistant. Based on the user's question, identify the distinct topics/aspects that need to be addressed and generate a separate search query for each one. Format your response as a list of search queries, one per line. Only output the search queries, nothing else.'''
        
        query_response = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': topic_identification_prompt},
                {'role': 'user', 'content': query},
            ]
        )
        
        search_queries = query_response['message']['content'].strip().split('\n')
        print('\nGenerated search queries:')
        for q in search_queries:
            print(f' - {q}')
        
        # Retrieve knowledge for each search query
        all_retrieved_knowledge = []
        for q in search_queries:
            retrieved = retrieve(q, VECTOR_DB, EMBEDDING_MODEL)
            all_retrieved_knowledge.extend(retrieved)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_retrieved_knowledge = []
        for chunk, similarity in all_retrieved_knowledge:
            if chunk not in seen:
                seen.add(chunk)
                unique_retrieved_knowledge.append((chunk, similarity))
        
        print('\nRetrieved knowledge:')
        for chunk, similarity in unique_retrieved_knowledge:
            print(f' - (similarity: {similarity:.2f}) {chunk}')
        
        # Generate response
        instruction_prompt = f'''You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Make sure to address all aspects of the question. Don't make up any new information:
        {'\n'.join([f' - {chunk}' for chunk, similarity in unique_retrieved_knowledge])}
        '''
        
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': query},
            ],
            stream=True,
        )
        
        print('\nChatbot response:')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        print("Make sure Ollama is running and the models are pulled. Try running:")
        print("ollama pull nomic-embed-text")
        print("ollama pull llama2")

def main():
    # Example usage
    pdf_path = input("Enter the path to your PDF file: ")
    output_file = "extracted_text.txt"
    
    # Extract text from PDF
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Save extracted text to file
    save_text_to_file(extracted_text, output_file)
    
    # Split text into chunks (you can adjust the chunk size as needed)
    text_chunks = [chunk.strip() for chunk in extracted_text.split('\n\n') if chunk.strip()]
    
    # Set up RAG system
    print("\nSetting up RAG system...")
    VECTOR_DB, EMBEDDING_MODEL, LANGUAGE_MODEL = setup_rag_system(text_chunks)
    
    # Interactive query loop
    while True:
        query = input("\nAsk a question about the PDF (or type 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        process_query(query, VECTOR_DB, EMBEDDING_MODEL, LANGUAGE_MODEL)

if __name__ == "__main__":
    main() 