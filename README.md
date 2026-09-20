# RAG From Scratch - Dental Knowledge Assistant

A modular Retrieval-Augmented Generation (RAG) system built from scratch for dental research and clinical documentation. This project enables intelligent querying of dental theses, research papers, and clinical documents using state-of-the-art embeddings and LLM technology.

## Table of Contents

- [RAG From Scratch - Dental Knowledge Assistant](#rag-from-scratch---dental-knowledge-assistant)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Use Cases](#key-use-cases)
  - [Features](#features)
  - [Architecture](#architecture)
  - [Tech Stack](#tech-stack)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Step 1: Clone the Repository](#step-1-clone-the-repository)
    - [Step 2: Create Virtual Environment](#step-2-create-virtual-environment)
    - [Step 3: Install Dependencies](#step-3-install-dependencies)
    - [Step 4: Configure Environment Variables](#step-4-configure-environment-variables)
    - [Step 5: Prepare Your Documents](#step-5-prepare-your-documents)
  - [Project Structure](#project-structure)
  - [Usage Guide](#usage-guide)
  - [Configuration](#configuration)
    - [Adjusting Chunk Size](#adjusting-chunk-size)
  - [API Reference](#api-reference)
  - [Acknowledgments](#acknowledgments)

---

## Overview

This RAG system is specifically designed for the dental domain, processing PDFs and text documents from dental theses and research papers. It provides a conversational interface for querying dental knowledge, with sources and references for every answer.

### Key Use Cases

- **Clinical Decision Support**: Query dental procedures, treatments, and best practices
- **Research Assistance**: Extract information from dental theses and studies
- **Educational Tool**: Learn about dental conditions, treatments, and protocols
- **Knowledge Management**: Centralize and search dental documentation

---

## Features

- **Multi-format Document Loading**: Supports PDF and text files
- **Intelligent Chunking**: Recursive text splitting with configurable chunk sizes
- **Semantic Embeddings**: Multilingual sentence transformers for accurate retrieval
- **Vector Storage**: Persistent ChromaDB storage with cosine similarity search
- **RAG Pipeline**: Clean, modular architecture with separate concerns
- **REST API**: FastAPI endpoint for programmatic access
- **Web Interface**: User-friendly HTML interface for querying
- **Source Attribution**: Every answer includes source references
- **Environment Config**: Secure API key management with `.env`

---

## Architecture
```text
┌───────────────────────────────────────────────────┐
│                     User Interface                │
│                (index.html / API clients)         │
└─────────────────────────┬─────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────┐
│                     API Layer                     │
│                     (api.py)                      │
│ ┌───────────────────────────────────────────────┐ │
│ │    POST /chat │ GET /health │ POST /index     │ │
│ └───────────────────────────────────────────────┘ │
└─────────────────────────┬─────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────┐
│                    RAG Pipeline                   │
│ ┌───────────────────────────────────────────────┐ │
│ │ 1. Document Loading (loader.py)               │ │
│ │ ├── PDF Loader (PyMuPDF)                      │ │
│ │ └── Text Loader (TextLoader)                  │ │
│ ├───────────────────────────────────────────────┤ │
│ │ 2. Chunking (splitter.py)                     │ │
│ │ └── RecursiveCharacterTextSplitter            │ │
│ ├───────────────────────────────────────────────┤ │
│ │ 3. Embedding (embedding.py)                   │ │
│ │ └── SentenceTransformer (Multilingual)        │ │
│ ├───────────────────────────────────────────────┤ │
│ │ 4. Vector Store (vector_store.py)             │ │
│ │ └── ChromaDB (Persistent)                     │ │
│ ├───────────────────────────────────────────────┤ │
│ │ 5. Retrieval (retriever.py)                   │ │
│ │ └── Similarity Search with filtering          │ │
│ └───────────────────────────────────────────────┘ │
└─────────────────────────┬─────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────┐
│ LLM Layer                                         │
│ (Groq / Qwen Model)                               │
│ ┌───────────────────────────────────────────────┐ │
│ │      Context + Question → LLM → Answer        │ │
│ └───────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Embedding Model** | SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2) | Latest |
| **Vector Database** | ChromaDB | Persistent |
| **LLM Provider** | Groq (Qwen 3.8-27B) | Latest |
| **Framework** | FastAPI | 0.115+ |
| **Document Processing** | LangChain | 0.3+ |
| **PDF Parsing** | PyMuPDF | 1.24+ |
| **Python** | Python | 3.9+ |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Groq API key (for LLM access)

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd RAG_from_scratch
```

### Step 2: Create Virtual Environment
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a .env file in the project root:

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```

### Step 5: Prepare Your Documents
Place your dental documents in the data/ directory:

```text
data/
├── text_files/          # For .txt files
│   ├── python_intro.txt
│   └── machine_learning.txt
└── pdf/                 # For PDF files
    ├── thesis1.pdf
    ├── thesis2.pdf
    └── ...
```

---

## Project Structure
```text
RAG_from_scratch/
├── .venv/                      # Virtual environment
├── data/                       # Data directory
│   ├── text_files/             # Text documents
│   ├── pdf/                    # PDF documents
│   └── vector_store/           # ChromaDB persistence (auto-created)
├── notebook/                   # Development notebooks
│   └── documents.ipynb
├── \__init__.py                 # Package initialization
├── .env                        # Environment variables (gitignored)
├── .gitignore                  # Git ignore file
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration settings
├── loader.py                   # Document loading functions
├── splitter.py                 # Text chunking functions
├── embedding.py                # Embedding generation
├── vector_store.py             # Vector database management
├── retriever.py                # Document retrieval logic
├── main.py                     # CLI entry point
├── api.py                      # FastAPI server
├── index.html                  # Web interface
└── README.md                   # This file
```
---

## Usage Guide
1. Index Your Documents
First, index all documents into the vector database:

```bash
python main.py
```
Expected Output:
```text
--- Indexation des documents ---
Loading PDF files from: data/pdf
Split 1483 documents into 4011 chunks
Loading embedding model: paraphrase-multilingual-MiniLM-L12-v2
Model loaded successfully. Embedding dimension: 384
Generating embeddings for 4011 texts...
Adding 4011 documents to vector store...
Successfully added 4011 documents to vector store
--- Indexation terminée ---
```

2. Start the API Server
```bash
python api.py
```
Expected Output:

```text
Loading embedding model: paraphrase-multilingual-MiniLM-L12-v2
Model loaded successfully. Embedding dimension: 384
Vector store initialized. Collection: pdf_documents
Existing documents in collection: 4011
RAG prêt !
INFO:     Uvicorn running on http://127.0.0.1:8000
```

3. Query via Web Interface
Open index.html in your browser and start asking questions about dental topics.

4. Query via Command Line
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended size for an oral mucosal biopsy?"}'
```

5. Query via Python
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"question": "What are the clinical features of oral lichen planus?"}
)

data = response.json()
print(f"Answer: {data['answer']}")
print("\nSources:")
for source in data['sources']:
    print(f"  - {source['source']}: {source['content']}")
```

---

## Configuration
All configuration parameters are centralized in config.py:

```python
# config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Text Splitter Configuration
CHUNK_SIZE = 1000              # Characters per chunk
CHUNK_OVERLAP = 200            # Overlap between chunks

# Loader Configuration
PDF_GLOB_PATTERN = "**/*.pdf"
TEXT_GLOB_PATTERN = "**/*.txt"
TEXT_ENCODING = "utf-8"

# Vector Store Configuration
COLLECTION_NAME = "pdf_documents"
VECTOR_STORE_PATH = "data/vector_store"

# Embedding Configuration
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# LLM Configuration
LLM_MODEL = "qwen/qwen3.8-27b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
```

### Adjusting Chunk Size
For dental documents with technical terminology, you may want to adjust chunk size:

```python
# config.py
CHUNK_SIZE = 1500    # Larger chunks for technical content
CHUNK_OVERLAP = 300  # More overlap for context preservation
```

---

## API Reference
`POST /chat`
Query the RAG system with a question.

Request Body:

```json
{
  "question": "string"
}
```
Response:

```json
{
  "answer": "string",
  "sources": [
    {
      "source": "filename.pdf",
      "content": "relevant excerpt..."
    }
  ]
}
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes peri-implantitis?"}'
```
`GET /health` (Optional)
Check API health status.

---

## Acknowledgments
- University of Geneva - For dental research context
- LangChain - For document processing tools
- ChromaDB - For vector storage
- Groq - For LLM inference
- SentenceTransformers - For embedding models