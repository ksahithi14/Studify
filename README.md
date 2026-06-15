# Studify: Multi-Document RAG Console

## 🛠️ Tech Stack

* **Orchestration Framework:** LangChain (LCEL)
* **Backend Engine:** FastAPI (Asynchronous ASGI server)
* **Presentation Layer:** Streamlit (Pure Python multi-page layout engine)
* **Vector Database:** ChromaDB (Persistent local embed storage)
* **Embeddings Model:** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
* **Inference Model:** Ollama (`llama3.1`) / OpenAI API

---

## 🚀 Core Functionalities

### 1. Advanced Ingestion Pipeline
* **Multi-Format Document Parsing:** Ingests complex PDFs dynamically, parsing operational text and layout properties into distinct document objects.
* **Intelligent Content Pruning:** Leverages a fast preprocessing pipeline to detect and discard structural noise (such as indices, tables of contents, and license boilerplates) before chunking, maximizing vector index hygiene.
* **Dynamic Metadata Injection:** Automatically signs every vectorized chunk with its explicit originating filename (`source_name`) and page context frames during database writes.

### 2. High-Precision Contextual Q&A
* **Multi-Tenant Domain Scoping:** The console detects which document is targeted by the user and locks downstream queries to that explicit file.
* **Isolated Subspace Retrieval:** Injects a strict metadata filter lock (`search_kwargs={"filter": {"source_name": target_document}}`) directly into the LangChain retrieval runtime, guaranteeing zero cross-document information leak or contamination.
* **Asynchronous Request Threading:** Routes user traffic across non-blocking async endpoint workers to prevent system execution stalls during heavy vector queries or long LLM completion windows.

---
<img width="1709" height="980" alt="image" src="https://github.com/user-attachments/assets/688873ce-5eec-4d3c-b9d5-9b7a4a7ad6a9" />

