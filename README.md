<div align="center">

![mypdfrag](assets/images/mypdfrag.png)

# mypdfrag

### Local-first PDF Intelligence. Simple, Lightweight, Private.

[![Tests](https://github.com/rafterkode/mypdfrag/actions/workflows/tests.yml/badge.svg)](https://github.com/rafterkode/mypdfrag/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey?logo=linux&logoColor=black)](https://www.linux.org/)
[![GitHub last commit](https://img.shields.io/github/last-commit/rafterkode/mypdfrag)](https://github.com/rafterkode/mypdfrag/commits/main)

A lightweight Retrieval-Augmented Generation (RAG) application for querying PDF documents using locally hosted language models.

[Features](#features) · [Architecture](#architecture) · [Installation](#installation) · [Usage](#usage)

</div>

---

## Overview

**mypdfrag** is a personal document assistant built with Python, Ollama, ChromaDB, and FastAPI.

It transforms PDF documents into a searchable knowledge base and uses locally hosted language models to generate context-aware answers with source references.

**Designed around three principles:**

- **Local-first:** Documents, embeddings, and queries stay on your machine.
- **Lightweight:** Designed to operate with modest hardware and without a dedicated GPU.
- **Private:** No external LLM API required.

## Features

| Feature | Description |
|:--|:--|
| PDF Ingestion | Extract and process PDF documents using PyMuPDF |
| Semantic Search | Retrieve relevant content using vector similarity |
| Local Embeddings | Generate embeddings with EmbeddingGemma |
| Local AI | Generate answers using Qwen 2.5 |
| Source References | Return document names and corresponding page numbers |
| REST API | Access the application through FastAPI |
| Persistent Storage | Store document embeddings in ChromaDB |

## Commercial Version

Looking for a ready-to-use Windows/macOS version with a graphical interface?

Check out the **full version on Gumroad**.

> The open-source version is fully functional. The commercial version adds convenience features for non-developer users.

---

## Architecture

mypdfrag uses a straightforward Retrieval-Augmented Generation pipeline.

### Technology Stack

| Component | Technology |
|:--|:--|
| LLM Runtime | Ollama |
| Language Model | Qwen 2.5 1.5B |
| Embedding Model | EmbeddingGemma 300M QAT Q4_0 |
| PDF Extraction | PyMuPDF |
| Vector Database | ChromaDB |
| API Framework | FastAPI |

### RAG Pipeline

```text
             DOCUMENT INGESTION
             ──────────────────

  PDF Documents
       │
       ▼
  Text Extraction (PyMuPDF)
       │
       ▼
  Text Chunking
       │
       ▼
  Embeddings (EmbeddingGemma)
       │
       ▼
  ChromaDB (Vector Database)


             QUESTION & ANSWER
             ─────────────────

  User Question
       │
       ▼
  Question Embedding
       │
       ▼
  Semantic Search ◄──── ChromaDB
       │
       ▼
  Relevant Context
       │
       ▼
  Ollama (Qwen 2.5)
       │
       ▼
  Generated Answer
       │
       ▼
  Answer + Source References
```

### Main Components

- **Ingestion:** Extracts, chunks, and indexes PDF content.
- **Embeddings:** Converts text into vector representations.
- **Retrieval:** Finds relevant document chunks using semantic similarity.
- **Generation:** Uses retrieved context to generate answers.
- **API:** Exposes the application through a local REST API.

---

## System Requirements

### Hardware

Designed to run on relatively modest hardware.

| Resource | Requirement |
|:--|:--|
| CPU | 4–8 cores |
| RAM | 8 GB |
| GPU | Not required |
| Operating System | Linux (Ubuntu / Debian) |

### Software

- Python 3.10+
- pip
- Python venv
- Ollama

---

## Installation

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify the installation:

```bash
ollama --version
systemctl status ollama
```

### 2. Download the Models

**Language model:**

```bash
ollama pull qwen2.5:1.5b
```

**Embedding model:**

```bash
ollama pull embeddinggemma:300m-qat-q4_0
```

Both models are downloaded and executed locally through Ollama.

### 3. Configure Ollama for Limited Resources

Create a systemd override:

```bash
sudo systemctl edit ollama
```

Add the following configuration:

```ini
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_CONTEXT_LENGTH=2048"
Environment="OLLAMA_KEEP_ALIVE=0"
```

Restart Ollama:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

This configuration prioritizes sequential model execution to reduce memory consumption.

### 4. Clone the Repository

```bash
git clone https://github.com/rafterkode/mypdfrag.git
cd mypdfrag
```

### 5. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 6. Install Dependencies

```bash
pip install -r requirements.txt
```

**Main Python dependencies:**

```text
fastapi
uvicorn
chromadb
pymupdf
requests
```

---

## Project Structure

```text
mypdfrag/
│
├── documents/
│   └── document1.pdf
│
├── database/
│   └── ChromaDB vector database
│
├── app/
│   ├── ingest.py
│   └── api.py
│
├── tests/
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

| Path | Description |
|:--|:--|
| `documents/` | PDF documents to be indexed |
| `database/` | Persistent ChromaDB storage |
| `app/ingest.py` | PDF extraction, chunking, and indexing |
| `app/api.py` | Search and answer generation API |
| `tests/` | Automated tests |
| `requirements.txt` | Python dependencies |

---

## Usage

### 1. Add PDF Documents

Place your PDF files inside:

```text
documents/
```

PDF files containing selectable text can be processed directly.

> Scanned PDFs require an additional OCR step, which is not included in the current version.

### 2. Index Documents

From the project root directory:

```bash
python app/ingest.py
```

The ingestion process:

1. Reads PDF documents.
2. Extracts text using PyMuPDF.
3. Splits extracted text into chunks.
4. Generates embeddings using Ollama.
5. Stores vectors and associated metadata in ChromaDB.

Indexing only needs to be performed when documents are added or modified.

### 3. Start the API

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8181
```

The API is now available locally.

| Resource | URL |
|:--|:--|
| API | http://127.0.0.1:8181 |
| Interactive documentation | http://127.0.0.1:8181/docs |

---

## Querying Documents

### API Endpoint

```http
POST /ask
```

### Example Request

Set the question as an environment variable:

```bash
QUESTION="What are the key points of the document?"
```

Send the request:

```bash
curl -X POST http://127.0.0.1:8181/ask \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg question "$QUESTION" '{question: $question}')"
```

Alternatively, send the question directly:

```bash
curl -X POST http://127.0.0.1:8181/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key points of the document?"
  }'
```

### Response

The API returns:

- The original question.
- The generated answer.
- The source documents used to generate the answer.
- The corresponding page numbers.

Answers are generated using document chunks retrieved from the vector database.

---

## Performance & Resource Management

mypdfrag is designed to operate on machines with limited hardware resources and no dedicated GPU.

### Resource Optimization

The following principles help limit resource consumption:

- Small language model.
- Separate embedding and generation models.
- One loaded model at a time.
- Context length limited to 2,048 tokens.
- Single parallel request.
- Model unloading after use.

Actual performance depends on CPU capabilities, document size, retrieved context, and database volume.

### Debugging & Performance Monitoring

Performance measurements are available in `app/api.py` through the `DEBUG` configuration variable.

Enable diagnostic output:

```python
DEBUG = True
```

This displays execution times for:

- Embedding generation.
- ChromaDB retrieval.
- Text generation.
- Total response time.

Disable diagnostic output:

```python
DEBUG = False
```

---

## Current Limitations

mypdfrag is currently a lightweight Retrieval-Augmented Generation proof of concept (POC).

Known limitations:

- Text generation is relatively slow because inference runs entirely on the CPU.
- Broad questions requiring comprehensive document summaries may exceed the capabilities of the current retrieval strategy.
- The number of retrieved chunks is currently fixed at 3.
- Long answers may require a higher generation token limit.
- Answer quality depends on the relevance and coverage of retrieved document chunks.
- No conversation history.

---

## Privacy & Security

mypdfrag follows a local-first approach.

- Documents are processed locally.
- Embeddings are generated locally.
- User queries are processed locally.
- No external LLM API is required after downloading the models and installing dependencies.
- The FastAPI server listens on `127.0.0.1` by default, limiting exposure to the local machine.

This architecture provides greater control over document processing and avoids sending document content to external LLM providers.

---

## License

Personal experimental project.

Attribution is appreciated and helps support the original project and its development.

The licenses of Ollama models and third-party dependencies remain applicable independently of this project's license.