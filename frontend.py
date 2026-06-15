# frontend.py
import streamlit as st
import requests

# FastAPI Endpoints
BASE_URL = "http://127.0.0.1:8000"
ASK_URL = f"{BASE_URL}/ask"
INGEST_URL = f"{BASE_URL}/ingest"
DOCS_URL = f"{BASE_URL}/documents"

st.set_page_config(page_title="Multi-Doc AI", page_icon="📚", layout="wide")

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.title("⚙️ Control Dashboard")
    st.subheader("1. Ingest New Documents")
    
    # File Uploader component
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Vectorize & Save File", use_container_width=True):
            with st.spinner("Processing document layout vectors..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(INGEST_URL, files=files)
                    if response.status_code == 200:
                        st.success(f"Successfully processed: {uploaded_file.name}")
                        st.rerun() # Refresh layout views to populate list updates
                    else:
                        st.error(f"Error handling file upload processing.")
                except Exception as e:
                    st.error(f"Connection failure: {str(e)}")
                    
    st.markdown("---")
    st.subheader("2. Select Target Scope")
    
    # Pull collection availability catalogs directly from our database indexer
    available_docs = []
    try:
        doc_response = requests.get(DOCS_URL, timeout=3)
        if doc_response.status_code == 200:
            available_docs = doc_response.json().get("documents", [])
    except Exception:
        st.warning("Database unavailable. Boot FastAPI app server first.")

    if available_docs:
        selected_doc = st.selectbox("Choose document context pipeline target:", available_docs)
        st.info(f"Target locked onto context paths of:\n**{selected_doc}**")
    else:
        selected_doc = None
        st.caption("⚠️ No documents detected in the database vector paths yet. Ingest a file above!")

# --- MAIN CHAT APPLICATION WINDOW ---
st.title("📚 Multi-Document Enterprise RAG Console")
st.write("Isolate conversations to an explicit file context path using local computation clusters entirely.")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Output chat history structures cleanly
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt ingestion handlers
if user_query := st.chat_input("Ask a question regarding the active targeted document scope...", disabled=(not selected_doc)):
    
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Isolating vector regions and querying context frames..."):
            try:
                payload = {
                    "question": user_query,
                    "document_name": selected_doc # Passing the explicit scope filter lock
                }
                response = requests.post(ASK_URL, json=payload)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "Empty output frames.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Engine parsing failure frame (Status: {response.status_code})")
            except Exception as e:
                st.error(f"Network error linking framework environments: {str(e)}")