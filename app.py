import streamlit as st
from src.pipeline import answer_query, get_source_chunks

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("PDF Chatbot (Local Embeddings)")

# Sidebar info
with st.sidebar:
    st.markdown("**Model:** TinyLlama-1.1B-Chat-v1.0")
    chunk_info = get_source_chunks("chunks", info_only=True)
    st.markdown(f"**Chunks:** {chunk_info['chunk_count']}")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

def clear_chat():
    st.session_state.history = []

st.button("Clear Chat", on_click=clear_chat)

try:
    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Generating answer..."):
            # Get answer as a generator for streaming
            answer_gen, sources = answer_query(query, "vectordb", stream=True, return_sources=True)
            answer_placeholder = st.empty()
            answer_text = ""
            for chunk in answer_gen:
                answer_text += chunk
                answer_placeholder.markdown(answer_text)
            st.session_state.history.append((query, answer_text, sources))
            st.markdown("**Source Chunks:**")
            for i, src in enumerate(sources):
                st.markdown(f"**Chunk {i+1}:** {src}")
except Exception as e:
    st.error(f"An error occurred: {e}")