# main.py
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TEXT_GLOB_PATTERN, PDF_GLOB_PATTERN, TEXT_ENCODING
from loader import load_text_files, load_pdf_files
from splitter import split_documents

def run_rag_pipeline():
    """
    Exécute le pipeline RAG de chargement et de split des documents.
    """
    print("--- Starting RAG Pipeline ---")
    
    # load text files
    text_dir = f"{DATA_DIR}/text_files"
    print(f"\nLoading text files from: {text_dir}")
    text_documents = load_text_files(text_dir, TEXT_GLOB_PATTERN, TEXT_ENCODING)
    
    # load PDF files
    pdf_dir = f"{DATA_DIR}/pdf"
    print(f"\nLoading PDF files from: {pdf_dir}")
    pdf_documents = load_pdf_files(pdf_dir, PDF_GLOB_PATTERN)

    # combine docs if needed
    all_documents = text_documents + pdf_documents
    print(f"\nTotal documents loaded: {len(all_documents)}")
    
    # Splitting combined docs
    print(f"\n--- Splitting documents ---")
    split_docs = split_documents(all_documents, CHUNK_SIZE, CHUNK_OVERLAP)
    
    print("\n--- Pipeline Complete ---")

    return split_docs

if __name__ == "__main__":
    # if the script is executed directly, launch the pipeline
    chunks = run_rag_pipeline()