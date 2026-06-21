# RAG Knowledge Engine Platform

<!-- Optional banner image -->
<!-- You can insert a screenshot or architecture image here -->

---

## 1. Project Overview

The RAG Knowledge Engine Platform is an end-to-end Retrieval-Augmented Generation (RAG) system that enables users to upload PDF documents and ask natural language questions about their content.

The system integrates semantic search, vector databases, and large language models to generate grounded and context-aware responses based on the uploaded documents.

---

## 2. Project Goal

The goal of this project is to build a production-style RAG system capable of:

- Processing and understanding PDF documents
- Performing semantic retrieval using vector search
- Generating accurate answers using a local LLM
- Providing simple and interactive user interfaces

---

## 3. Problem Statement

Traditional document search systems rely on keyword-based matching, which fails to capture semantic meaning and context.

This project solves this limitation by using embedding-based retrieval and vector similarity search to find relevant information, enabling intelligent question answering over unstructured documents.

---

## 4. Core Idea

The core idea of this system is Retrieval-Augmented Generation (RAG), which combines information retrieval with large language models:

1. Documents are split into semantic chunks
2. Chunks are converted into embeddings
3. FAISS is used to perform similarity search
4. Relevant context is added to the prompt
5. A language model generates grounded answers

---

## 5. System Overview

The system follows a modular architecture:

- Users interact via Streamlit or Gradio interfaces
- Requests are sent to a FastAPI backend
- The backend manages ingestion, retrieval, and LLM inference
- FAISS handles vector similarity search
- Ollama (Llama 3) generates responses

---

## 6. Key Components

The system is composed of modular components that work together to enable Retrieval-Augmented Generation (RAG):

### Backend (FastAPI)
- Handles PDF ingestion
- Processes document chunking
- Generates embeddings
- Manages FAISS vector search
- Builds prompts for the LLM
- Exposes REST API endpoints

### Retrieval Layer
- FAISS vector database for similarity search
- SentenceTransformers for embedding generation
- Chunk-level metadata tracking (page references)

### Language Model
- Ollama runtime for local inference
- Llama 3 model for response generation

### Frontend Interfaces
- Streamlit: primary user interface for document QA
- Gradio: lightweight alternative interface for testing and interaction

### Infrastructure
- Docker for containerization
- Docker Compose for multi-service orchestration

---

## 7. What You Will Learn

This project demonstrates practical implementation of modern AI engineering concepts:

- Retrieval-Augmented Generation (RAG) systems
- Vector databases and semantic search (FAISS)
- Embedding generation using transformer models
- Local LLM deployment with Ollama
- API design using FastAPI
- Full-stack ML system architecture
- Multi-container deployment with Docker Compose
- Separation of frontend and backend services in ML systems

---

## 8. System Overview (High-Level Architecture)

The system is designed as a modular pipeline:

```
User
│
▼
Streamlit / Gradio UI
│
▼
FastAPI Backend
│
▼
RAG Pipeline Engine
│
├──────────────┬──────────────┐
▼ ▼
FAISS SentenceTransformers
(Vector DB) (Embeddings)
│
└────── Retrieval ──────┘
│
▼
Context-Augmented Prompt
│
▼
Ollama (Llama 3)
│
▼
Generated Answer + Sources
```

---

## 9. Data Flow Explanation

The system processes a user query through the following pipeline:

1. User uploads a PDF document.
2. The document is split into smaller text chunks.
3. Each chunk is converted into embeddings using SentenceTransformers.
4. Embeddings are stored in a FAISS vector index.
5. When a question is asked:
   - The query is embedded
   - FAISS retrieves the most similar chunks
6. Retrieved context is combined with the user question
7. The prompt is sent to the LLM (Llama 3 via Ollama)
8. The model generates a context-aware response with citations

---

## 10. Project Structure

```
rag-knowledge-engine-platform/
│
├── app/
│ ├── api/ # FastAPI endpoints
│ ├── rag/ # RAG pipeline logic
│ ├── retrieval/ # FAISS retrieval logic
│ ├── ingestion/ # PDF processing & chunking
│
├── gradio/ # Gradio frontend
├── ui/streamlit/ # Streamlit frontend
├── data/ # PDF storage and processed data
│
├── Dockerfile
├── Dockerfile.streamlit
├── Dockerfile.gradio
├── docker-compose.yml
│
├── requirements.txt
├── pyproject.toml
└── README.md

```


---

## 11. Technology Stack

### Backend
- FastAPI
- Uvicorn

### Retrieval System
- FAISS
- SentenceTransformers
- PyPDF2

### LLM Layer
- Ollama
- Llama 3

### Frontend
- Streamlit
- Gradio

### DevOps / Infrastructure
- Docker
- Docker Compose

### Language
- Python 3.10+

---

## 12. DevOps & Infrastructure

The project is fully containerized and designed for reproducible execution.

### Key DevOps Features

- Multi-container orchestration using Docker Compose
- Isolated services for backend and frontends
- Environment-based configuration using `.env`
- Local LLM deployment using Ollama
- Consistent runtime across machines

### Services Overview

- `rag-api`: FastAPI backend service
- `rag-streamlit`: Streamlit UI service
- `rag-gradio`: Gradio UI service
- `ollama`: local LLM runtime (external service)

---

## 13. Design Principles

The system was designed following key engineering principles:

- **Separation of Concerns**: Backend, retrieval, and UI layers are fully decoupled
- **Modularity**: Each component (ingestion, embedding, retrieval, generation) is independent
- **Reproducibility**: Docker ensures consistent execution across environments
- **Scalability-ready design**: Architecture can be extended to multi-document systems
- **Local-first AI**: Uses Ollama for private, offline LLM inference

---

## 14. Core Features

- PDF document upload and processing
- Semantic chunking with page-level tracking
- Vector-based search using FAISS
- Context-aware question answering
- Source citation with page references
- Dual frontend support (Streamlit + Gradio)
- Fully containerized deployment

---

## 15. Supported Use Cases

This system can be used for:

- 📄 Document Question Answering
- 📚 Academic paper exploration
- 🧠 Knowledge base search
- 📊 Internal document assistants
- 🏢 Enterprise document retrieval systems
- 🤖 LLM-powered research assistants

---

## 16. RAG Pipeline Summary

The system implements a standard Retrieval-Augmented Generation workflow:

- Documents are embedded into vector space
- Semantic similarity search retrieves relevant chunks
- Retrieved context is injected into the LLM prompt
- The model generates grounded responses based on retrieved data

This ensures responses are **context-aware and document-grounded**, reducing hallucinations.

---

## 17. RAG Pipeline (Detailed View)

This section provides a deeper breakdown of the Retrieval-Augmented Generation pipeline.

### Step 1: Document Processing
- PDF files are uploaded via the UI
- Text is extracted page by page
- Content is cleaned and normalized

### Step 2: Chunking Strategy
- Documents are split into semantic chunks
- Each chunk retains metadata (page number, source file)

### Step 3: Embedding Generation
- Each chunk is converted into a vector representation
- Model used:
  - sentence-transformers/all-MiniLM-L6-v2

### Step 4: Vector Storage
- FAISS index stores embeddings for fast similarity search

### Step 5: Retrieval
- User query is embedded
- FAISS retrieves top-k similar chunks

### Step 6: Prompt Construction
- Retrieved chunks are combined with the user query
- A structured prompt is created for the LLM

### Step 7: LLM Inference
- Ollama (Llama 3) processes the prompt
- Generates a grounded response

### Step 8: Response Augmentation
- Final answer includes:
  - Generated response
  - Source document references
  - Page-level citations

---

## 18. Docker Architecture (Final View)

The system is fully containerized and orchestrated using Docker Compose.

### Services

- **rag-api** → FastAPI backend (RAG engine)
- **rag-streamlit** → Streamlit UI
- **rag-gradio** → Gradio UI
- **ollama** → External LLM runtime (Llama 3)

### Architecture Flow

```
User Interface (Streamlit / Gradio)
│
▼
FastAPI Backend (rag-api)
│
▼
RAG Processing Layer
│
┌───────┴────────┐
▼ ▼
FAISS Embeddings
(Vector DB) (SentenceTransformers)
│
└──── Retrieval ────┘
│
▼
Ollama (Llama 3)
│
▼
Final Answer + Citations

```

---

## 19. Installation & Environment Setup

### Prerequisites
- Python 3.10+
- Docker
- Docker Compose
- Ollama installed locally

### Clone Repository

```
git clone https://github.com/<your-username>/rag-knowledge-engine-platform.git
cd rag-knowledge-engine-platform

```

---

### Create Environment


conda create -n llm-env python=3.10 -y
conda activate llm-env


---

### Install Dependencies


pip install -r requirements.txt


---

### Pull LLM Model (Ollama)


ollama pull llama3


Start Ollama:


ollama serve


---

### Environment Variables

Create a `.env` file in the project root:


MODEL_NAME=llama3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

---

## 20. Docker Setup + Run Commands (Compose Flow)

This section explains how to build and run the entire system using Docker Compose.

---

### Step 1: Build Docker Images

Build all required service images:


docker build -t rag-api .

docker build -t rag-streamlit -f Dockerfile.streamlit .

docker build -t rag-gradio -f Dockerfile.gradio .


---

### Step 2: Start the Full System (Docker Compose)

Run all services together:


docker compose up


This will start:

- FastAPI backend (rag-api)
- Streamlit UI
- Gradio UI
- Ollama connection (external)

---

### Step 3: Access the Services

Once running, open:

- FastAPI → http://localhost:8000
- FastAPI Docs → http://localhost:8000/docs
- Streamlit → http://localhost:8501
- Gradio → http://localhost:7860

---

### Step 4: Stop the System

To safely stop all running containers:


CTRL + C


Then optionally clean up:


docker compose down


---

### Notes

- If containers are already built, `docker compose up` is enough.
- Rebuild images only if Dockerfiles or dependencies change.
- Ollama runs outside Docker and must be started separately:


ollama serve

---

## 21. Streamlit Interface

The Streamlit interface provides a user-friendly web application for interacting with the RAG system.

It allows users to:

- Upload PDF documents
- Process and index documents
- Ask natural language questions
- View generated answers with context

---

### Run Streamlit Application


streamlit run ui/streamlit/app.py


---

### Access Interface

Once the application is running, open:


http://localhost:8501


---

### Features

- Simple and interactive UI
- File upload support (PDF)
- Real-time question answering
- Displays model-generated responses
- Lightweight and fast for experimentation

---

### Notes

- Streamlit connects directly to the FastAPI backend.
- Ensure the FastAPI server is running before using the interface.
- If Docker Compose is used, Streamlit will already be available automatically.

---

## 22. Gradio Interface

The Gradio interface provides an alternative lightweight UI for interacting with the RAG system.

It is mainly used for:

- Quick testing of the RAG pipeline
- Uploading PDF documents
- Asking questions directly
- Visualizing responses in a simple interface

---

### Run Gradio Application


python gradio/app.py


---

### Access Interface

Once running, open:


http://localhost:7860


---

### Features

- Minimal and fast UI for experimentation
- Supports PDF upload and question answering
- Direct integration with FastAPI backend
- Useful for debugging and quick validation

---

### Notes

- Gradio runs independently from Streamlit
- Both interfaces can run simultaneously if needed
- Ensure FastAPI backend is running before using Gradio

---

## 23. Testing Strategy

The system was validated using a multi-layer testing approach to ensure correctness across all components.

---

### 1. Backend API Testing

- FastAPI endpoints were tested using REST requests
- Verified:
  - `/upload` correctly processes PDF files
  - `/ask` returns relevant answers
  - Error handling for invalid inputs

Example test:

```
Invoke-RestMethod http://127.0.0.1:8000/ask -Method POST
-ContentType "application/json" `
-Body '{"question":"What is supervised learning?"}'
```

---

### 2. RAG Pipeline Testing

- Verified document ingestion and chunking
- Checked embedding generation consistency
- Validated FAISS retrieval accuracy
- Ensured LLM responses were grounded in retrieved context

---

### 3. Docker Testing

- All services tested using Docker Compose:
  - FastAPI backend
  - Streamlit UI
  - Gradio UI
  - Ollama integration

Verified:
- Container communication
- Service startup order
- Port accessibility

---

### 4. UI Testing

#### Streamlit
- PDF upload functionality
- Question-answer interaction
- UI responsiveness

#### Gradio
- Lightweight interaction testing
- Fast feedback loop for queries

---

### 5. Integration Testing

- End-to-end pipeline tested:
  1. Upload document
  2. Index embeddings
  3. Ask question
  4. Retrieve context
  5. Generate answer via LLM

---

### Key Observation

The system was validated in both:
- Local environment
- Docker Compose environment

Ensuring consistency across deployments.
---
## 24. Key Insights & Lessons Learned

This project provided important practical insights into building and deploying end-to-end AI systems.

---

### 1. Working with Local LLMs is Resource Intensive

- Running Llama 3 locally through Ollama requires significant RAM
- System stability is highly dependent on memory management
- Improper load can lead to system freezes or slowdowns

---

### 2. Docker Compose Simplifies Complex Systems

- Managing multiple services manually is error-prone
- Docker Compose allows orchestration of:
  - Backend API
  - Multiple UIs
  - External LLM service
- Single-command deployment improves reliability

---

### 3. Vector Search is Critical for RAG Quality

- FAISS provides fast and efficient similarity search
- Retrieval quality directly impacts LLM output quality
- Chunking strategy significantly affects performance

---

### 4. Proper Chunking Matters More Than Expected

- Poor chunking leads to irrelevant retrieval
- Page-level metadata improves traceability
- Balanced chunk size improves both recall and precision

---

### 5. Separation of Concerns Improves Maintainability

- Splitting backend, retrieval, and UI layers improves clarity
- Easier debugging and independent testing
- Enables scalable system design

---

### 6. System Stability Depends on Resource Management

- Running multiple services simultaneously can cause memory spikes
- Docker containers need careful orchestration in low-resource environments
- Avoid unnecessary parallel testing during heavy workloads

---

## 25. Phase B Roadmap (Next Development Phase)

This project is the foundation of a larger system that will be extended in Phase B.

Phase B is already planned as the next stage of development and will focus on evolving the current RAG system into a more advanced and scalable architecture.

---

### Planned Direction for Phase B

- Support for multi-document ingestion
- Migration from FAISS to ChromaDB for persistent storage
- Improved retrieval strategies with metadata filtering
- Enhanced evaluation of retrieval and generation quality
- Optimization of memory usage and performance
- Potential cloud deployment (AWS / Azure)

---

### Key Idea

Phase B is not a theoretical extension — it is the next iteration of this system, aimed at improving scalability, robustness, and production readiness.

---

## 26. Final Takeaways

This project demonstrates the end-to-end implementation of a Retrieval-Augmented Generation (RAG) system that integrates modern AI and software engineering practices.

Key achievements include:

- Building a complete RAG pipeline from document ingestion to LLM response generation
- Implementing semantic search using FAISS and transformer-based embeddings
- Integrating a local LLM (Llama 3 via Ollama) for inference
- Designing a modular backend using FastAPI
- Creating multiple user interfaces (Streamlit and Gradio)
- Containerizing the entire system using Docker and Docker Compose

---

### Engineering Perspective

This project goes beyond a simple AI demo and represents a production-style architecture with:

- Separation of concerns
- Scalable service design
- Reproducible deployment
- Real-world RAG implementation patterns

---

### Conclusion

The system provides a solid foundation for building advanced AI applications over private or enterprise document data, and serves as a base for future scaling and improvements.

---

## 27. Author

AI Engineering project built for portfolio and production-style demonstration.

Aspiring AI / Machine Learning Engineer focused on building end-to-end AI systems.

Built and experimented with:

- RAG-based systems
- LangChain-style LLM pipelines
- Ollama (local LLM inference)
- Llama 3 integration
- LLM application architecture design
- Docker-based deployment and orchestration
- FastAPI backend systems for AI applications
- Streamlit and Gradio interfaces

🔗 GitHub: https://github.com/Javier-DataScience