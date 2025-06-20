from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
from typing import Optional
import re

def clean_text(text: str) -> str:
    """
    Clean text by removing headers, footers, page numbers, and extra whitespace.
    Customize the regex patterns as needed for your document.
    """
    # Remove lines that are just numbers (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove repeated header/footer (customize this pattern)
    text = re.sub(r'YourHeaderText|YourFooterText', '', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Strip leading/trailing whitespace
    return text.strip()

def chunk_pdf(pdf_path: str, chunks_dir: str, chunk_size: int = 500, 
              chunk_overlap: int = 50, save_metadata: bool = True):
    """
    Chunk a PDF file and save chunks to directory.
    
    Args:
        pdf_path: Path to the PDF file
        chunks_dir: Directory to save chunks
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        save_metadata: Whether to save metadata alongside chunks
    """
    try:
        # Load PDF
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages")
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        docs = splitter.split_documents(pages)
        
        # Create output directory
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Save chunks and metadata
        for i, doc in enumerate(docs):
            chunk_filename = f"chunk_{i:04d}.txt"
            chunk_path = os.path.join(chunks_dir, chunk_filename)
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(clean_text(doc.page_content))
            if save_metadata:
                metadata_filename = f"chunk_{i:04d}_metadata.json"
                metadata_path = os.path.join(chunks_dir, metadata_filename)
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(doc.metadata, f, indent=2)
        
        print(f"Successfully saved {len(docs)} chunks to {chunks_dir}")
        return len(docs)
        
    except FileNotFoundError:
        print(f"Error: PDF file not found: {pdf_path}")
        return 0
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return 0

def get_chunk_info(chunks_dir: str) -> dict:
    """Get information about existing chunks in directory."""
    if not os.path.exists(chunks_dir):
        return {"chunk_count": 0, "total_size": 0}
    
    chunk_files = [f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".txt")]
    total_size = sum(os.path.getsize(os.path.join(chunks_dir, f)) for f in chunk_files)
    
    return {
        "chunk_count": len(chunk_files),
        "total_size": total_size,
        "chunks_dir": chunks_dir
    }

if __name__ == "__main__":
    pdf_file = "data/AI_Training_Document.pdf"
    output_dir = "chunks"
    
    # Check if PDF exists
    if not os.path.exists(pdf_file):
        print(f"PDF file not found: {pdf_file}")
        print("Please ensure the PDF file exists in the data directory.")
    else:
        # Chunk the PDF
        chunk_count = chunk_pdf(pdf_file, output_dir, chunk_size=1200, chunk_overlap=200)
        
        # Display summary
        if chunk_count > 0:
            info = get_chunk_info(output_dir)
            print(f"\nSummary:")
            print(f"- Chunks created: {info['chunk_count']}")
            print(f"- Total size: {info['total_size']:,} bytes")
            print(f"- Output directory: {info['chunks_dir']}")