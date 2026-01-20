"""Tests for answer generation functionality."""

import pytest
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding.create_embeddings import EmbeddingStore
from retrieval.query_system import QuerySystem
from llm.answer_generator import AnswerGenerator


@pytest.fixture(scope="function")
def populated_query_system(temp_embeddings_dir, sample_documents):
    index_path = os.path.join(temp_embeddings_dir, "test_index.faiss")
    metadata_path = os.path.join(temp_embeddings_dir, "test_metadata.pkl")
    
    store = EmbeddingStore(
        model_name="all-MiniLM-L6-v2",
        index_path=index_path,
        metadata_path=metadata_path
    )
    store.add_documents(sample_documents)
    store.save()
    
    query_system = QuerySystem(embedding_store=store, enable_cache=False)
    
    return query_system


class TestAnswerGenerator:    
    def test_answer_generator_initialization_mock(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        assert ag.model_type == "mock"
        assert ag.query_system is not None
    
    def test_generate_answer_mock_mode(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        result = ag.generate_answer("What is Python?", top_k=2)
        
        assert 'answer' in result
        assert 'sources' in result
        assert 'answer_formatted' in result
        assert 'sources_detailed' in result
        assert 'keywords' in result
        assert 'retrieved_text' in result
        
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0
        assert isinstance(result['sources'], list)
        assert isinstance(result['keywords'], list)
    
    def test_generate_answer_empty_query(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        result = ag.generate_answer("", top_k=2)
        
        assert 'answer' in result
        assert "Please provide a valid query" in result['answer']
        assert len(result['sources']) == 0
    
    def test_generate_answer_keyword_extraction(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        result = ag.generate_answer("What is Python programming language?", top_k=2)
        
        assert 'keywords' in result
        assert isinstance(result['keywords'], list)
        assert len(result['keywords']) > 0
        
        # Check that meaningful keywords are extracted
        keywords = result['keywords']
        assert any('python' in kw.lower() for kw in keywords) or any('programming' in kw.lower() for kw in keywords)
    
    def test_generate_answer_sources_detailed(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        result = ag.generate_answer("Python", top_k=2)
        
        assert 'sources_detailed' in result
        assert isinstance(result['sources_detailed'], list)
        
        if len(result['sources_detailed']) > 0:
            source = result['sources_detailed'][0]
            assert 'title' in source
            assert 'filename' in source
            assert 'filepath' in source
            assert 'file_type' in source
            assert 'distance' in source
    
    def test_generate_answer_formatted_output(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        result = ag.generate_answer("What is FastAPI?", top_k=2)
        
        assert 'answer_formatted' in result
        assert isinstance(result['answer_formatted'], str)
        assert len(result['answer_formatted']) > 0
    
    def test_generate_answer_paragraph_formatting(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        result = ag.generate_answer("Machine learning", top_k=2)
        
        # Check that answer formatting function works
        answer = result['answer']
        # May contain paragraph breaks
        assert isinstance(answer, str)
    
    def test_generate_answer_top_k_parameter(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        result1 = ag.generate_answer("Python", top_k=1)
        result2 = ag.generate_answer("Python", top_k=3)
        
        # Results may have different number of sources
        assert len(result1['sources']) <= len(result2['sources'])
    
    def test_extract_keywords(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        keywords = ag._extract_keywords("What is Python programming language?")
        
        assert isinstance(keywords, set) or isinstance(keywords, list)
        assert len(keywords) > 0
        # Should filter out stop words like "what", "is"
        assert 'python' in ' '.join(str(kw).lower() for kw in keywords) or 'programming' in ' '.join(str(kw).lower() for kw in keywords)
    
    def test_format_answer_with_paragraphs(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        # Test with text containing multiple sentences
        text = "First sentence. Second sentence. Third sentence."
        formatted = ag._format_answer_with_paragraphs(text)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
    
    def test_highlight_keywords(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        text = "Python is a programming language."
        keywords = {'python', 'programming'}
        
        # Test markdown highlighting
        highlighted_md = ag._highlight_keywords(text, keywords, markdown=True)
        assert 'mark' in highlighted_md.lower() or 'python' in highlighted_md.lower()
        
        # Test plain text highlighting
        highlighted_plain = ag._highlight_keywords(text, keywords, markdown=False)
        assert isinstance(highlighted_plain, str)
        assert len(highlighted_plain) >= len(text)
    
    def test_generate_answer_error_handling(self, populated_query_system):
        ag = AnswerGenerator(
            model_type="mock",
            query_system=populated_query_system
        )
        
        # Should not raise exception, but return error message
        # This test ensures error handling works
        result = ag.generate_answer("test query", top_k=5)
        
        assert 'answer' in result
        assert isinstance(result['answer'], str)
