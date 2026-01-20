
import pytest
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding.create_embeddings import EmbeddingStore
from retrieval.query_system import QuerySystem


@pytest.fixture(scope="function")
def populated_embedding_store(temp_embeddings_dir, sample_documents):
    index_path = os.path.join(temp_embeddings_dir, "test_index.faiss")
    metadata_path = os.path.join(temp_embeddings_dir, "test_metadata.pkl")
    
    store = EmbeddingStore(
        model_name="all-MiniLM-L6-v2",
        index_path=index_path,
        metadata_path=metadata_path
    )
    store.add_documents(sample_documents)
    store.save()
    
    return store, index_path, metadata_path


class TestQuerySystem:
    
    def test_query_system_initialization(self, populated_embedding_store):
        store, index_path, metadata_path = populated_embedding_store
        
        query_system = QuerySystem(embedding_store=store, enable_cache=False)
        
        assert query_system.embedding_store is not None
        assert query_system.model is not None
        assert query_system.snippet_length == 500
    
    def test_query_documents_basic(self, populated_embedding_store):
        store, index_path, metadata_path = populated_embedding_store
        
        query_system = QuerySystem(embedding_store=store, enable_cache=False)
        
        results = query_system.query_documents("Python programming", top_k=2)
        
        assert len(results) > 0
        assert len(results) <= 2
        assert all('title' in result for result in results)
        assert all('filename' in result for result in results)
        assert all('snippet' in result for result in results)
        assert all('distance' in result for result in results)
    
    def test_query_documents_top_k(self, populated_embedding_store):
        store, index_path, metadata_path = populated_embedding_store
        
        query_system = QuerySystem(embedding_store=store, enable_cache=False)
        
        # Request top_k=1
        results = query_system.query_documents("Python", top_k=1)
        assert len(results) == 1
        
        # Request top_k=3 (should get all documents if we have 3)
        results = query_system.query_documents("Python", top_k=3)
        assert len(results) <= 3
    
    def test_query_documents_empty_query(self, populated_embedding_store):
        store, index_path, metadata_path = populated_embedding_store
        
        query_system = QuerySystem(embedding_store=store, enable_cache=False)
        
        results = query_system.query_documents("", top_k=5)
        assert len(results) == 0
        
        results = query_system.query_documents("   ", top_k=5)
        assert len(results) == 0
    
    def test_query_documents_cache(self, populated_embedding_store):
        store, index_path, metadata_path = populated_embedding_store
        
        query_system = QuerySystem(embedding_store=store, enable_cache=True, cache_size=10)
        
        query = "Python programming language"
        
        # First query - should be cache miss
        results1 = query_system.query_documents(query, top_k=2)
        stats1 = query_system.get_cache_stats()
        assert stats1['cache_misses'] == 1
        assert stats1['cache_hits'] == 0
        
        # Second query - should be cache hit
        results2 = query_system.query_documents(query, top_k=2)
        stats2 = query_system.get_cache_stats()
        assert stats2['cache_hits'] == 1
        assert stats2['cache_misses'] == 1
        
        # Results should be identical
        assert len(results1) == len(results2)
        assert results1[0]['title'] == results2[0]['title']
    
    def test_query_documents_cache_disabled(self, populated_embedding_store):
        store, index_path, metadata_path = populated_embedding_store
        
        query_system = QuerySystem(embedding_store=store, enable_cache=False)
        
        query = "Python"
        query_system.query_documents(query, top_k=2)
        query_system.query_documents(query, top_k=2)
        
        stats = query_system.get_cache_stats()
        assert stats['cache_enabled'] is False
    
    def test_clear_cache(self, populated_embedding_store):
        store, index_path, metadata_path = populated_embedding_store
        
        query_system = QuerySystem(embedding_store=store, enable_cache=True)
        
        # Add some queries to cache
        query_system.query_documents("Python", top_k=2)
        query_system.query_documents("FastAPI", top_k=2)
        
        stats_before = query_system.get_cache_stats()
        assert stats_before['cache_size'] > 0
        
        # Clear cache
        query_system.clear_cache()
        
        stats_after = query_system.get_cache_stats()
        assert stats_after['cache_size'] == 0
        assert stats_after['cache_hits'] == 0
        assert stats_after['cache_misses'] == 0
    
    def test_get_cache_stats(self, populated_embedding_store):
        store, index_path, metadata_path = populated_embedding_store
        
        query_system = QuerySystem(embedding_store=store, enable_cache=True)
        
        stats = query_system.get_cache_stats()
        
        assert 'cache_enabled' in stats
        assert 'cache_size' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'hit_rate_percent' in stats
        assert stats['cache_enabled'] is True
    
    def test_query_documents_result_structure(self, populated_embedding_store):
        store, index_path, metadata_path = populated_embedding_store
        
        query_system = QuerySystem(embedding_store=store, enable_cache=False)
        
        results = query_system.query_documents("Python", top_k=1)
        
        if len(results) > 0:
            result = results[0]
            
            # Check all required fields
            assert 'index' in result
            assert 'title' in result
            assert 'filename' in result
            assert 'filepath' in result
            assert 'file_type' in result
            assert 'snippet' in result
            assert 'distance' in result
            
            # Check types
            assert isinstance(result['index'], int)
            assert isinstance(result['title'], str)
            assert isinstance(result['filename'], str)
            assert isinstance(result['distance'], float)
