"""Streamlit Web UI for RAG system"""

import sys
from pathlib import Path
import streamlit as st
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_chain import RAGChain
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Chat System",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 RAG Chat System")
st.markdown("""This is a Retrieval-Augmented Generation (RAG) system powered by Ollama and Pinecone.""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Initialize config
    config = Config()
    
    # Display current configuration
    st.subheader("Current Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunk Size", config.CHUNK_SIZE)
        st.metric("LLM Model", config.OLLAMA_MODEL)
    with col2:
        st.metric("Top-K", config.TOP_K)
        st.metric("Embedding Model", config.EMBEDDING_MODEL.split('/')[-1])
    
    st.divider()
    
    # Document upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Initialize RAG
                    rag = RAGChain(config)
                    
                    # Save uploaded files
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = config.DOCUMENTS_DIR / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(str(file_path))
                    
                    # Add documents
                    rag.add_documents(file_paths)
                    st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)")
                    
                    # Store RAG in session
                    st.session_state.rag = rag
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # System statistics
    st.subheader("📊 System Statistics")
    try:
        if hasattr(st.session_state, 'rag'):
            rag = st.session_state.rag
            stats = rag.get_stats()
            st.json(stats)
        else:
            st.info("Upload and process documents to see statistics")
    except Exception as e:
        st.warning(f"Could not fetch statistics: {str(e)}")

# Main content
if not hasattr(st.session_state, 'rag'):
    try:
        config = Config()
        st.session_state.rag = RAGChain(config)
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        st.stop()

rag = st.session_state.rag

# Chat interface
st.header("💬 Chat")

# User input
user_question = st.text_input(
    "Ask your question:",
    placeholder="What would you like to know?"
)

if user_question:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        show_sources = st.checkbox("Show sources", value=True)
    
    with col2:
        if st.button("Ask", type="primary"):
            with st.spinner("Generating response..."):
                try:
                    if show_sources:
                        result = rag.query_with_sources(
                            user_question,
                            top_k=5,
                            return_sources=True
                        )
                        
                        st.subheader("📝 Answer")
                        st.write(result['answer'])
                        
                        if result['sources']:
                            st.subheader("📚 Sources")
                            for i, source in enumerate(result['sources'], 1):
                                with st.expander(f"Source {i} (Similarity: {source['similarity_score']:.4f})"):
                                    st.write(source['content'])
                                    st.caption(f"Metadata: {source['metadata']}")
                    else:
                        answer = rag.query(user_question)
                        st.subheader("📝 Answer")
                        st.write(answer)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
---
**RAG Chat System** | Powered by Ollama + Pinecone + LangChain

[GitHub](https://github.com/VincentForTech/AI) • [Documentation](https://github.com/VincentForTech/AI)
""")
