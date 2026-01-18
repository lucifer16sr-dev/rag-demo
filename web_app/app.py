"""
Streamlit Web App for RAG Knowledge Assistant.

This module provides a web interface for interacting with the RAG-based
knowledge assistant system using Streamlit.

Installation:
    To run this app, install Streamlit:
    pip install streamlit

Usage:
    streamlit run web_app/app.py
"""

import streamlit as st
from llm.answer_generator import generate_answer

# Set page configuration
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .sources-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📚 RAG Knowledge Assistant</h1>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info(
        "This is a RAG (Retrieval-Augmented Generation) based knowledge assistant. "
        "Enter your questions below to get answers based on your document knowledge base."
    )
    st.header("Instructions")
    st.write("1. Enter your question in the text box below")
    st.write("2. Click 'Get Answer' button")
    st.write("3. View the generated answer and sources")

# Main content area
st.markdown("---")

# Query input
query = st.text_input(
    "Enter your question:",
    placeholder="What would you like to know?",
    key="query_input"
)

# Get Answer button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submit_button = st.button("Get Answer", type="primary", use_container_width=True)

# Process query when button is clicked
if submit_button:
    if query and query.strip():
        # Show loading spinner while generating answer
        with st.spinner("Generating answer... This may take a moment."):
            try:
                # Generate answer
                result = generate_answer(query, top_k=3)
                
                # Display answer
                st.markdown("### Answer:")
                st.markdown(f'<div class="answer-box">{result.get("answer", "No answer generated.")}</div>', 
                           unsafe_allow_html=True)
                
                # Display sources
                sources = result.get("sources", [])
                if sources:
                    st.markdown("### Sources:")
                    st.markdown('<div class="sources-box">', unsafe_allow_html=True)
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"{i}. {source}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No sources available.")
            
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {str(e)}")
                st.info("Please try again or check if your embeddings are set up correctly.")
    else:
        st.warning("Please enter a valid query.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "RAG Knowledge Assistant | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
