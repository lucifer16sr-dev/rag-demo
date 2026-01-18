
import streamlit as st
import sys
import os
sys.path.insert(0, os.getcwd())

from llm.answer_generator import AnswerGenerator, generate_answer

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
        background-color: #f8f9fa !important;
        color: #262730 !important;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
        line-height: 1.8;
        font-size: 1rem;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .answer-box * {
        color: #262730 !important;
    }
    .answer-box p {
        margin-bottom: 1em;
    }
    .sources-box {
        background-color: #e8f4f8;
        color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border: 1px solid #c0d9e0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📚 RAG Knowledge Assistant</h1>', unsafe_allow_html=True)

# Initialize answer generator in session state (cache it)
if 'answer_generator' not in st.session_state:
    with st.spinner("Initializing answer generator (this may take a moment on first load)..."):
        try:
            # Use mock mode for faster responses, or auto to try OpenAI/HuggingFace
            st.session_state.answer_generator = AnswerGenerator(model_type="mock")
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Failed to initialize answer generator: {str(e)}")
            st.session_state.initialized = False

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
    
    if st.session_state.get('initialized', False):
        st.success("✓ System ready")
    else:
        st.warning("⚠ System not initialized")

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
        if not st.session_state.get('initialized', False):
            st.error("Answer generator not initialized. Please refresh the page.")
        else:
            # Show loading spinner while generating answer
            with st.spinner("Generating answer... This may take a moment."):
                try:
                    # Generate answer using cached generator
                    result = st.session_state.answer_generator.generate_answer(query, top_k=3)
                    
                    # Display answer
                    st.markdown("### Answer:")
                    answer_text = result.get("answer", "No answer generated.")
                    # Format answer with better readability - replace double newlines with paragraphs
                    formatted_answer = answer_text.replace('\n\n', '\n\n')
                    # Display in a styled container
                    st.markdown(f'<div class="answer-box">{formatted_answer}</div>', 
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
                    import traceback
                    st.error(f"An error occurred while generating the answer: {str(e)}")
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
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
