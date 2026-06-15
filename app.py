# app.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import shutil
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_core.documents import Document

from main import build_rag_chain, PERSIST_DIR, COLLECTION_NAME

app = FastAPI(title="Multi-Document Local RAG API")
vectorstore = None  # Persistent global vector store reference

class QueryRequest(BaseModel):
    question: str
    document_name: str  # The user selects which file to filter by

@app.on_event("startup")
def startup_event():
    """Initializes the persistent Chroma instance on startup."""
    global vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize the store (will create the folder if it doesn't exist yet)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    print("🚀 Persistent Vector Database is loaded and operational.")

@app.get("/documents")
def list_documents():
    """Scans Chroma metadata structures to return a list of unique uploaded files."""
    global vectorstore
    try:
        # Fetch records from database
        data = vectorstore.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        # Pull out unique names saved under our custom source_name attribute key
        unique_files = sorted(list(set(m.get("source_name") for m in metadatas if m and "source_name" in m)))
        return {"documents": unique_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest_uploaded_document(file: UploadFile = File(...)):
    """Receives a binary file over HTTP, processes chunks, and labels them."""
    global vectorstore
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported currently.")
    
    # Save a temporary copy locally to read
    temp_path = Path(f"./temp_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Extract text using your main logic flow
        reader = PdfReader(str(temp_path))
        docs = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source_name": file.filename,  # Storing plain file identifier
                        "page": page_num
                    }
                ))
        
        if not docs:
            raise ValueError("The uploaded PDF appears to contain no readable machine text layout.")
            
        # Split chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        chunks = splitter.split_documents(docs)
        
        # Inject records directly into our persistent running engine store instance
        vectorstore.add_documents(chunks)
        return {"status": "success", "message": f"Successfully vectorized {len(chunks)} text chunks from '{file.filename}'"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion processing failure: {str(e)}")
    finally:
        if temp_path.exists():
            os.remove(temp_path) # Clean up file system buffer tracks

@app.post("/ask")
def ask_document(payload: QueryRequest):
    global vectorstore
    try:
        # Compile a dynamic chain explicitly locked down to this specific file's metadata source block
        dynamic_chain = build_rag_chain(vectorstore, target_document=payload.document_name)
        result = dynamic_chain.invoke({"input": payload.question})
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))