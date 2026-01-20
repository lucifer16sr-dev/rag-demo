
import pytest
import os
import shutil
from pathlib import Path
import tempfile

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "sample_documents"


@pytest.fixture(scope="session")
def test_data_dir():
    return TEST_DATA_DIR


@pytest.fixture(scope="function")
def temp_embeddings_dir():
    temp_dir = tempfile.mkdtemp()
    embeddings_dir = Path(temp_dir) / "test_embeddings"
    embeddings_dir.mkdir(parents=True)
    
    yield str(embeddings_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_documents():
    return [
        {
            'title': 'test_doc1',
            'filename': 'test_doc1.md',
            'filepath': str(TEST_DATA_DIR / 'test_doc1.md'),
            'content': 'Python is a high-level programming language known for its simplicity and readability.',
            'file_type': 'markdown',
            'language': 'en'
        },
        {
            'title': 'test_doc2',
            'filename': 'test_doc2.md',
            'filepath': str(TEST_DATA_DIR / 'test_doc2.md'),
            'content': 'FastAPI is a modern, fast web framework for building APIs with Python.',
            'file_type': 'markdown',
            'language': 'en'
        },
        {
            'title': 'test_doc3',
            'filename': 'test_doc3.md',
            'filepath': str(TEST_DATA_DIR / 'test_doc3.md'),
            'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data.',
            'file_type': 'markdown',
            'language': 'en'
        }
    ]
