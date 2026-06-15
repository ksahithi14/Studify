# pip install pypdf chromadb langchain langchain-community langchain-text-splitters langchain-openai

from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import json
from langchain_core.messages import HumanMessage
import os


PDF_PATH = "/Users/kasahith/2RAG/pg69504-images-3.pdf"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_chunks"


def extract_pdf_pages(pdf_path: str) -> list[Document]:
    """Extract text from each page of a PDF as LangChain Documents."""
    reader = PdfReader(pdf_path)
    docs: list[Document] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_name": str(Path(pdf_path).resolve()),
                        "page": page_num,
                    },
                )
            )

    return docs

def is_useful_content(chunk_text: str, llm: ChatOllama) -> bool:
    """Uses a local LLM to classify if a chunk contains actual book/document content."""
    
    prompt = (
        "You are an expert document classifier. Analyze the following text snippet from a document. "
        "Determine if it contains meaningful informational content (like narrative, data, chapters, analysis) "
        "OR if it is purely structural noise (like a Table of Contents, Index, legal disclaimers, licensing agreements, "
        "copyright pages, or blank page placeholders).\n\n"
        f"Text Snippet:\n{chunk_text[:1000]}\n\n"
        "Respond strictly in JSON format with a single key 'is_content' and a boolean value (true or false).\n"
        "Example: {\"is_content\": true}"
    )
    
    try:
        # We use a stream or direct invoke to get a quick classification
        response = llm.invoke([HumanMessage(content=prompt)])
        # Clean up potential LLM markdown wrapping
        clean_res = response.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_res)
        return data.get("is_content", True)
    except Exception:
        # If the LLM formatting fails, err on the side of caution and keep the data
        return True

def build_vector_store(pdf_path: str) -> Chroma:
    # 1) Extract page text
    page_docs = extract_pdf_pages(pdf_path)

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=400)
    chunks = splitter.split_documents(page_docs)

    
    # 4) Filter chunks dynamically
    filter_llm = ChatOllama(model="llama3.1", temperature=0, format="json")

    print(f"Analyzing {len(chunks)} chunks for structural noise...")
    cleaned_chunks = []
    for i, chunk in enumerate(chunks):
        if is_useful_content(chunk.page_content, filter_llm):
            cleaned_chunks.append(chunk)
        else:
            print(f"Skipped noise chunk {i+1} (Likely index, TOC, or license)")

    print(f"Retained {len(cleaned_chunks)} / {len(chunks)} chunks.")

    # 3) Embeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4) Store in Chroma
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    vectorstore.add_documents(cleaned_chunks)
    return vectorstore

def build_rag_chain(vectorstore: Chroma, target_document: str = None):
    search_kwargs = {"k": 12}
    if target_document:
        search_kwargs["filter"] = {"source_name": target_document}
        
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict, precise document assistant. Answer the user's question using ONLY the provided context."
        ),
        (
            "human",
            "Context:\n{context}\n\n"
            "Question: {input}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer the question using ONLY the context provided above.\n"
            "2. If the similarity of extracted text is high, and answers the question you may provide a longer answer that can be written for 10 marks."
        ),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    return rag_chain


if __name__ == "__main__":
    # Check if database already exists so we don't duplicate data
    db_exists = os.path.exists(PERSIST_DIR)
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not db_exists:
        print("Building vector store for the first time...")
        vectorstore = build_vector_store(PDF_PATH)
    else:
        print("Loading existing vector store...")
        # Simply load the existing directory without re-adding documents
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )


    rag_chain = build_rag_chain(vectorstore)

    query = "Describe the history, methods, and types of ships used in the Arctic whale fishery."
    result = rag_chain.invoke({"input": query})

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nRETRIEVED CHUNKS:\n")
    for doc in result["context"]:
        print(f"Page {doc.metadata.get('page')}:")
        print(doc.page_content[:400])
        print("-" * 80)