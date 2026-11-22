![Tests](https://github.com/rafterkode/mypdfrag/actions/workflows/tests.yml/badge.svg)
<div align="center">

![mypdfrag](assets/images/LocalPDFRAG.png)

# mypdfrag

**A lightweight, local-first Retrieval-Augmented Generation (RAG) application for querying PDF documents using locally hosted language models.**

mypdfrag is a personal document assistant built with Python, Ollama, ChromaDB, and FastAPI.

It extracts text from PDF documents, generates embeddings, stores vector representations, and retrieves relevant content to generate context-aware answers using a locally running Large Language Model (LLM).

The main goal is to provide a fully local document assistant with limited hardware resource requirements, without relying on external LLM APIs.

---

## Commercial Version

Need a ready-to-use Windows/Mac version with a graphical interface ?

Check out the **full version** on Gumroad.

> The open-source version (this repository) is fully functional. The commercial version adds convenience features for non-developer users.

---

## 1. Technical Architecture

### Technology Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| LLM Runtime | Ollama | Local model execution |
| Language Model | Qwen 2.5 1.5B | Answer generation |
| Embedding Model | EmbeddingGemma 300M QAT Q4_0 | Text vectorization |
| PDF Extraction | PyMuPDF | PDF text extraction |
| Vector Database | ChromaDB | Vector storage and semantic search |
| API Framework | FastAPI | REST API endpoints |

### RAG Pipeline

```text
                  PDF DOCUMENTS
                       |
                       v
                  TEXT EXTRACTION
                     PyMuPDF
                       |
                       v
                   TEXT CHUNKING
                       |
                       v
                    EMBEDDINGS
              EmbeddingGemma 300M
                       |
                       v
                    CHROMADB
                  Vector Database

USER QUESTION
      |
      v
QUESTION EMBEDDING
      |
      v
SEMANTIC SEARCH <--------- CHROMADB
      |
      v
RELEVANT CONTEXT
      |
      v
OLLAMA / QWEN 2.5
      |
      v
GENERATED ANSWER
      |
      v
ANSWER WITH SOURCE REFERENCES
```

### Main Components

- **Ingestion:** Extracts, chunks, and indexes PDF content.

- **Embeddings:** Converts text into vector representations.

- **Retrieval:** Finds relevant document chunks using semantic similarity.

- **Generation:** Uses the retrieved context to generate answers.

- **API:** Exposes the application through a local REST API.

---

## 2. System Requirements

### Target Hardware

The application is designed to run on relatively modest hardware.

- Linux (Ubuntu / Debian).

- 4 to 8 CPU cores.

- 8 GB RAM.

- No dedicated GPU required.

### Software Requirements

- Python 3.10 or higher.

- pip.

- Python venv.

- Ollama.

---

## 3. Installation

### Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify the installation:

```bash
ollama --version
systemctl status ollama
```

### Download the Models

Language model:

```bash
ollama pull qwen2.5:1.5b
```

Embedding model:

```bash
ollama pull embeddinggemma:300m-qat-q4_0
```

Both models are downloaded and executed locally through Ollama.

### Configure Ollama for Limited Resource Usage

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

Restart the service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

This configuration prioritizes sequential model execution to reduce memory consumption.

---

## 4. Project Setup

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd mypdfrag
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

### Python Dependencies

```text
fastapi
uvicorn
chromadb
pymupdf
requests
```

---

## 5. Project Structure

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
|── tests/
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

### Directory Description

| Path | Description |
| --- | --- |
| `documents/` | PDF documents to be indexed |
| `database/` | Persistent ChromaDB vector storage |
| `app/ingest.py` | PDF extraction, chunking, and indexing |
| `app/api.py` | Search and answer generation API |
| `requirements.txt` | Python dependencies |

---

## 6. Usage

### Add PDF Documents

Place your PDF files inside:

```text
documents/
```

PDF files containing selectable text can be processed directly.

Scanned PDFs require an additional OCR step, which is not included in the current version.

### Index Documents

From the project root directory, run:

```bash
python app/ingest.py
```

The ingestion process performs the following operations:

1. Reads PDF documents.

2. Extracts text using PyMuPDF.

3. Splits the extracted text into chunks.

4. Generates embeddings using Ollama.

5. Stores vectors and associated metadata in ChromaDB.

Indexing only needs to be performed when new documents are added or existing documents are modified.

### Start the API

Run the FastAPI application:

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8181
```

The API is available locally at:

```text
http://127.0.0.1:8181
```

Interactive API documentation:

```text
http://127.0.0.1:8181/docs
```

---

## 7. Querying Documents

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

Alternatively, replace the question directly in the JSON payload:

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

Answers are generated using the document chunks retrieved from the vector database.

---

## 8. Performance and Resource Management

The project is designed to operate on a machine with limited hardware resources and no dedicated GPU.

### Resource Optimization

The following principles are used to limit resource consumption:

- Use of a small language model.

- Separate embedding and generation models.

- One loaded model at a time.

- Context length limited to 2,048 tokens.

- Single parallel request.

- Model unloading after use.

Actual performance depends on CPU capabilities, document size, retrieved context, and database volume.

### Debugging and Performance Monitoring

Performance measurements are available in `app/api.py` through the `DEBUG` configuration variable.

Enable diagnostic output:

```python
DEBUG = True
```

This displays the execution time of each processing step:

- Embedding generation.

- ChromaDB retrieval.

- Text generation.

- Total response time.

Disable diagnostic output:

```python
DEBUG = False
```

---

## 9. Current Limitations

This project is currently a lightweight Retrieval-Augmented Generation proof of concept (POC).

### Known Limitations

- Text generation is relatively slow because inference runs entirely on the CPU.

- Broad questions requiring comprehensive document summaries may exceed the capabilities of the current retrieval strategy.

- The number of retrieved chunks is currently fixed at 3.

- Long answers may require a higher generation token limit.

- Answer quality depends on the relevance and coverage of retrieved document chunks.

- No conversation history.

---
## 10. Privacy and Security

mypdfrag is designed with a local-first approach.

Documents, embeddings, and user queries are processed locally.

After downloading the models and installing the dependencies, no external LLM API is required to operate the application.

The FastAPI server listens on `127.0.0.1` by default, limiting its exposure to the local machine.

This architecture provides greater control over document processing and avoids sending document content to external LLM providers.

---

## 11. License

Personal experimental project.\
Attribution is appreciated and helps support the original project and its development.

The licenses of the Ollama models and third-party dependencies remain applicable independently of this project's license.
