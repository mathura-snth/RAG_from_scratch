import os

# Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Text splitter
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Loader
PDF_GLOB_PATTERN = "**/*.pdf"
TEXT_GLOB_PATTERN = "**/*.txt"

# Encoding for text files
TEXT_ENCODING = "utf-8"

# Vector Store
COLLECTION_NAME = "pdf_documents"
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store")

# Embedding
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"