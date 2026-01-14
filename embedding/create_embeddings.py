
from pathlib import Path
from typing import List, Dict, Optional, Any
import pickle

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        f"Required packages not installed. Install with: pip install sentence-transformers faiss-cpu numpy. "
        f"Original error: {e}"
    )

from ingestion.ingest_documents import ingest_all_documents


class EmbeddingStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: Optional[int] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Infer embedding dimension from model if not provided
        if embedding_dim is None:
            # Get embedding dimension by encoding a test sentence
            test_embedding = self.model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
        else:
            self.embedding_dim = embedding_dim
        
        # Set default paths
        if index_path is None:
            index_path = "embeddings/vector_index.faiss"
        if metadata_path is None:
            metadata_path = "embeddings/metadata.pkl"
        
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        # Initialize FAISS index (L2 distance)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        
        # Create embeddings directory if it doesn't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _initialize_index(self):
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
    
    def add_documents(
        self,
        documents: List[Dict[str, str]],
        batch_size: int = 32
    ):
        if not documents:
            print("No documents provided to add.")
            return
        
        # Extract content for embedding
        contents = [doc['content'] for doc in documents]
        
        # Generate embeddings in batches
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(
            contents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Ensure embeddings are float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        # Initialize index if needed
        self._initialize_index()
        
        # Add embeddings to FAISS index
        self.index.add(embeddings)
        
        # Store metadata for each document
        for doc in documents:
            metadata_entry = {
                'title': doc.get('title', ''),
                'filename': doc.get('filename', ''),
                'filepath': doc.get('filepath', ''),
                'file_type': doc.get('file_type', ''),
            }
            self.metadata.append(metadata_entry)
        
        print(f"Added {len(documents)} documents to the vector database.")
        print(f"Total documents in database: {self.index.ntotal}")
    
    def save(self):
        if self.index is None:
            raise ValueError("No index to save. Add documents first.")
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        print(f"Saved FAISS index to {self.index_path}")
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Saved metadata to {self.metadata_path}")
    
    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))
        print(f"Loaded FAISS index from {self.index_path}")
        print(f"Index contains {self.index.ntotal} vectors")
        
        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"Loaded metadata for {len(self.metadata)} documents")
    
    def get_index_size(self):
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def get_metadata(self, index: int):
        if 0 <= index < len(self.metadata):
            return self.metadata[index]
        return None


def create_embeddings_from_documents(
    data_dir: str = "data",
    model_name: str = "all-MiniLM-L6-v2",
    index_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    save: bool = True
):
    print("=" * 60)
    print("Creating Embeddings for Documents")
    print("=" * 60)
    
    # Ingest documents
    print(f"\nStep 1: Ingesting documents from '{data_dir}'...")
    documents = ingest_all_documents(data_dir)
    
    if not documents:
        raise ValueError(f"No documents found in {data_dir}. Please add documents first.")
    
    # Create embedding store
    print(f"\nStep 2: Initializing embedding model '{model_name}'...")
    store = EmbeddingStore(
        model_name=model_name,
        index_path=index_path,
        metadata_path=metadata_path
    )
    
    # Add documents to store
    print(f"\nStep 3: Generating embeddings and adding to vector database...")
    store.add_documents(documents)
    
    # Save if requested
    if save:
        print(f"\nStep 4: Saving vector database to disk...")
        store.save()
    
    print("\n" + "=" * 60)
    print("Embedding creation completed successfully!")
    print(f"Total documents embedded: {store.get_index_size()}")
    print("=" * 60)
    
    return store


def load_embedding_store(
    model_name: str = "all-MiniLM-L6-v2",
    index_path: Optional[str] = None,
    metadata_path: Optional[str] = None
):
    store = EmbeddingStore(
        model_name=model_name,
        index_path=index_path,
        metadata_path=metadata_path
    )
    store.load()
    return store


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Check if user wants to create new embeddings or load existing
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        print("Loading existing embedding store...")
        store = load_embedding_store()
        print(f"Loaded {store.get_index_size()} documents")
    else:
        print("Creating new embeddings...")
        store = create_embeddings_from_documents()
        print(f"Created embeddings for {store.get_index_size()} documents")
