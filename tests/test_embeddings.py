
import pytest
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding.create_embeddings import EmbeddingStore, load_documents, create_embeddings_from_documents
from ingestion.ingest_documents import ingest_all_documents


class TestEmbeddingStore:
    
    def test_embedding_store_initialization(self, temp_embeddings_dir):
        index_path = os.path.join(temp_embeddings_dir, "test_index.faiss")
        metadata_path = os.path.join(temp_embeddings_dir, "test_metadata.pkl")
        
        store = EmbeddingStore(
            model_name="all-MiniLM-L6-v2",  # Use smaller model for faster tests
            index_path=index_path,
            metadata_path=metadata_path
        )
        
        assert store.model is not None
        assert store.embedding_dim > 0
        assert store.index_path == Path(index_path)
        assert store.metadata_path == Path(metadata_path)
    
    def test_add_documents(self, temp_embeddings_dir, sample_documents):
        index_path = os.path.join(temp_embeddings_dir, "test_index.faiss")
        metadata_path = os.path.join(temp_embeddings_dir, "test_metadata.pkl")
        
        store = EmbeddingStore(
            model_name="all-MiniLM-L6-v2",
            index_path=index_path,
            metadata_path=metadata_path
        )
        
        # Add documents
        store.add_documents(sample_documents)
        
        # Verify documents were added
        assert store.get_index_size() == len(sample_documents)
        assert len(store.metadata) == len(sample_documents)
        
        # Check metadata content
        for i, doc in enumerate(sample_documents):
            metadata = store.get_metadata(i)
            assert metadata is not None
            assert metadata['title'] == doc['title']
            assert metadata['filename'] == doc['filename']
    
    def test_save_and_load(self, temp_embeddings_dir, sample_documents):
        index_path = os.path.join(temp_embeddings_dir, "test_index.faiss")
        metadata_path = os.path.join(temp_embeddings_dir, "test_metadata.pkl")
        
        # Create and save
        store1 = EmbeddingStore(
            model_name="all-MiniLM-L6-v2",
            index_path=index_path,
            metadata_path=metadata_path
        )
        store1.add_documents(sample_documents)
        store1.save()
        
        # Verify files were created
        assert Path(index_path).exists()
        assert Path(metadata_path).exists()
        
        # Load into new store
        store2 = EmbeddingStore(
            model_name="all-MiniLM-L6-v2",
            index_path=index_path,
            metadata_path=metadata_path
        )
        store2.load()
        
        # Verify loaded data
        assert store2.get_index_size() == len(sample_documents)
        assert len(store2.metadata) == len(sample_documents)
        assert store2.get_index_size() == store1.get_index_size()
    
    def test_get_metadata(self, temp_embeddings_dir, sample_documents):
        index_path = os.path.join(temp_embeddings_dir, "test_index.faiss")
        metadata_path = os.path.join(temp_embeddings_dir, "test_metadata.pkl")
        
        store = EmbeddingStore(
            model_name="all-MiniLM-L6-v2",
            index_path=index_path,
            metadata_path=metadata_path
        )
        store.add_documents(sample_documents)
        
        # Test valid index
        metadata = store.get_metadata(0)
        assert metadata is not None
        assert metadata['title'] == sample_documents[0]['title']
        
        # Test invalid index
        metadata_invalid = store.get_metadata(100)
        assert metadata_invalid is None


class TestLoadDocuments:
    
    def test_load_documents_from_folder(self, test_data_dir):
        documents = load_documents(str(test_data_dir))
        
        assert len(documents) > 0
        assert all('title' in doc for doc in documents)
        assert all('content' in doc for doc in documents)
        assert all('filepath' in doc for doc in documents)
    
    def test_load_documents_handles_errors(self, tmp_path):
        non_existent_dir = tmp_path / "non_existent"
        documents = load_documents(str(non_existent_dir))
        
        # Should return empty list, not raise exception
        assert isinstance(documents, list)
        assert len(documents) == 0


class TestCreateEmbeddingsFromDocuments:
    
    def test_create_embeddings_from_documents(self, test_data_dir, temp_embeddings_dir):
        index_path = os.path.join(temp_embeddings_dir, "test_index.faiss")
        metadata_path = os.path.join(temp_embeddings_dir, "test_metadata.pkl")
        
        store = create_embeddings_from_documents(
            data_dir=str(test_data_dir),
            model_name="all-MiniLM-L6-v2",
            index_path=index_path,
            metadata_path=metadata_path,
            save=True
        )
        
        assert store is not None
        assert store.get_index_size() > 0
        assert Path(index_path).exists()
        assert Path(metadata_path).exists()
    
    def test_create_embeddings_empty_folder(self, tmp_path, temp_embeddings_dir):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        index_path = os.path.join(temp_embeddings_dir, "test_index.faiss")
        metadata_path = os.path.join(temp_embeddings_dir, "test_metadata.pkl")
        
        with pytest.raises(ValueError, match="No documents found"):
            create_embeddings_from_documents(
                data_dir=str(empty_dir),
                model_name="all-MiniLM-L6-v2",
                index_path=index_path,
                metadata_path=metadata_path
            )
