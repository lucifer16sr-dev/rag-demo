# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) based knowledge assistant system that enables you to query your documents using natural language. The system uses semantic search to find relevant information and generates answers based on your document knowledge base.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Configuration](#configuration)

## ✨ Features

- **Document Ingestion**: Load PDF and Markdown files from a folder
- **Vector Embeddings**: Create embeddings using state-of-the-art models (all-mpnet-base-v2)
- **Semantic Search**: Retrieve top-k most relevant documents using FAISS vector database
- **Answer Generation**: Generate answers using OpenAI, Hugging Face, or mock mode
- **Multiple Interfaces**: 
  - Command-line interface (CLI)
  - Streamlit web application
- **Advanced Features**:
  - Query caching for faster responses
  - Language detection
  - Keyword highlighting
  - Formatted answer display with paragraphs
  - Detailed source attribution

## 🔧 Installation

### Requirements

- Python 3.10 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd rag_knowledge_assistant
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

3. **Prepare your documents:**
   - Place your PDF or Markdown files in the `data/` folder

4. **Create embeddings:**
   ```bash
   python embedding/create_embeddings.py
   ```

This will:
- Load all PDF and Markdown files from the `data/` folder
- Generate embeddings using the `all-mpnet-base-v2` model
- Create a vector database in the `embeddings/` folder
- Save metadata for all documents

## 🚀 Quick Start

### 1. Prepare Your Documents

Add your documents to the `data/` folder:
```bash
data/
  ├── document1.pdf
  ├── document2.md
  └── document3.pdf
```

### 2. Create Embeddings

```bash
python embedding/create_embeddings.py
```

You should see output like:
```
============================================================
Creating Embeddings for Documents
============================================================

Step 1: Loading and cleaning documents from 'data'...
Step 2: Initializing embedding model 'all-mpnet-base-v2'...
Step 3: Generating embeddings and adding to vector database...
Step 4: Saving vector database to disk...

Embedding creation completed successfully!
Total documents embedded: 3
```

### 3. Run the Application

**CLI Mode:**
```bash
python cli/run_assistant.py
```

**Web App Mode:**
```bash
streamlit run web_app/app.py
```

## 💻 Usage

### Command-Line Interface (CLI)

Run the CLI assistant:
```bash
python cli/run_assistant.py
```

**Example Session:**
```
============================================================
Welcome to RAG Knowledge Assistant!
============================================================

Type your questions below. Type 'exit', 'quit', or 'q' to exit.
------------------------------------------------------------

Query: What is Python?

============================================================
ANSWER
============================================================
Python is a high-level programming language known for its 
simplicity and readability. It was created by Guido van 
Rossum and first released in 1991.

============================================================
SOURCES
============================================================
1. test_doc1 (test_doc1.md)
   Type: markdown | Relevance: 0.1234
   📄 data/test_doc1.md

============================================================
```

**Features:**
- Interactive query loop
- Loading indicators during processing
- Formatted answers with paragraph breaks
- Keyword highlighting (yellow background)
- Detailed source information with file paths
- Exit with 'exit', 'quit', or 'q'

### Streamlit Web Application

Run the web app:
```bash
streamlit run web_app/app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- Clean, modern web interface
- Text input for queries
- Real-time answer generation
- Formatted display with:
  - Paragraph-structured answers
  - Keyword highlighting
  - Source files with links
  - Keyword extraction display
- Sidebar with instructions
- Error handling and loading indicators

**Screenshots:**

*[Screenshot: Streamlit Web App - Main Interface]*
*[Screenshot: Streamlit Web App - Answer Display with Sources]*

## 📝 Example Queries and Outputs

### Example 1: Basic Query
**Query:** `What is Python?`

**Expected Output:**
```
Answer: Python is a high-level programming language known for 
its simplicity and readability. It was created by Guido van 
Rossum and first released in 1991.

Python supports multiple programming paradigms including 
object-oriented, imperative, functional, and procedural 
programming.

Sources:
1. Python Basics (python_basics.pdf)
2. Python Intermediate (python_intermediate.pdf)

Keywords: python, language, programming
```

### Example 2: Framework-Specific Query
**Query:** `How do I use FastAPI?`

**Expected Output:**
```
Answer: FastAPI is a modern, fast web framework for building 
APIs with Python. Install FastAPI using pip:
pip install fastapi uvicorn

Create a simple FastAPI application:
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

Sources:
1. FastAPI Backend (python_fastapi_backend.pdf)
```

### Example 3: Conceptual Query
**Query:** `What is machine learning?`

**Expected Output:**
```
Answer: Machine learning is a subset of artificial intelligence 
that enables computers to learn and make decisions from data 
without being explicitly programmed.

Types include:
- Supervised Learning: Learning from labeled training data
- Unsupervised Learning: Finding patterns in unlabeled data
- Reinforcement Learning: Learning through trial and error

Sources:
1. Machine Learning Basics (ml_basics.pdf)
```

## 📁 Project Structure

```
rag_knowledge_assistant/
├── data/                          # Place your documents here
│   ├── *.pdf                     # PDF documents
│   └── *.md                      # Markdown documents
│
├── embeddings/                    # Generated vector database
│   ├── vector_index.faiss        # FAISS vector index
│   └── metadata.pkl              # Document metadata
│
├── embedding/                     # Embedding module
│   ├── __init__.py
│   └── create_embeddings.py      # Embedding creation and storage
│
├── ingestion/                     # Document ingestion module
│   ├── __init__.py
│   └── ingest_documents.py       # PDF/Markdown parsing
│
├── retrieval/                     # Retrieval module
│   ├── __init__.py
│   └── query_system.py           # Query processing and caching
│
├── llm/                          # Answer generation module
│   ├── __init__.py
│   └── answer_generator.py       # LLM integration and formatting
│
├── cli/                          # Command-line interface
│   ├── __init__.py
│   └── run_assistant.py          # CLI application
│
├── web_app/                      # Streamlit web application
│   ├── __init__.py
│   └── app.py                    # Web interface
│
├── tests/                        # Test suite
│   ├── conftest.py               # Test fixtures
│   ├── test_embeddings.py        # Embedding tests
│   ├── test_retrieval.py         # Retrieval tests
│   ├── test_answer_generator.py  # Answer generation tests
│   └── sample_documents/         # Test documents
│
├── requirement.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🧪 Testing

Run the test suite using pytest:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_embeddings.py
pytest tests/test_retrieval.py
pytest tests/test_answer_generator.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=embedding --cov=retrieval --cov=llm --cov=ingestion
```

## ⚙️ Configuration

### Embedding Model

Default model: `all-mpnet-base-v2` (768 dimensions)

To use a different model, modify:
```python
# In embedding/create_embeddings.py
model_name = "your-preferred-model"
```

### Cache Configuration

Query caching is enabled by default with a cache size of 100 entries.

To disable or modify:
```python
# In retrieval/query_system.py
query_system = QuerySystem(enable_cache=True, cache_size=100)
```

### Answer Generation Model

The system supports three modes:
1. **OpenAI**: Requires `OPENAI_API_KEY` environment variable
2. **Hugging Face**: Uses local transformers models
3. **Mock**: Extracts relevant sentences (default, no API required)

Set the mode:
```python
# In llm/answer_generator.py or cli/web_app
ag = AnswerGenerator(model_type="mock")  # or "openai", "huggingface"
```

## 🔍 How It Works

1. **Document Ingestion**: 
   - Loads PDF and Markdown files from `data/` folder
   - Detects language and cleans text
   - Skips files with errors (logs warnings)

2. **Embedding Creation**:
   - Uses `all-mpnet-base-v2` model to create vector embeddings
   - Stores embeddings in FAISS vector database
   - Saves metadata (title, filename, filepath, file_type)

3. **Query Processing**:
   - Encodes user query into embedding vector
   - Searches FAISS index for top-k similar documents
   - Retrieves document snippets

4. **Answer Generation**:
   - Formats retrieved context
   - Generates answer using selected LLM
   - Formats with paragraphs and highlights keywords
   - Returns answer with source attribution

## 📊 Features in Detail

### Query Caching
- LRU cache for repeated queries
- Configurable cache size
- Cache statistics available
- Significant speedup for repeated queries

### Language Detection
- Automatically detects document language
- Uses `langdetect` library
- Language metadata stored with documents

### Keyword Highlighting
- Extracts keywords from queries
- Highlights keywords in answers
- Yellow background in web app
- ANSI colors in CLI

### Source Attribution
- Lists all source documents
- Includes file paths and relevance scores
- Displays document types
- Provides clickable links in web app

## 🐛 Troubleshooting

### No documents found
**Issue**: Error when creating embeddings
**Solution**: Ensure PDF or Markdown files are in the `data/` folder

### Import errors
**Issue**: ModuleNotFoundError
**Solution**: 
```bash
pip install -r requirement.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)  # Linux/Mac
set PYTHONPATH=%CD%                   # Windows
```

### Slow query responses
**Solution**: 
- Ensure embeddings are created
- Use query caching (enabled by default)
- Reduce `top_k` parameter for faster results

### Streamlit app not opening
**Solution**: 
- Check if port 8501 is available
- Manually open `http://localhost:8501`
- Check terminal for error messages

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📞 Support

For issues or questions, please open an issue on the project repository.

---

**Built with ❤️ using Python, Streamlit, FAISS, and Sentence Transformers**
