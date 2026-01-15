
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        f"Required packages not installed. Install with: pip install sentence-transformers faiss-cpu numpy. "
        f"Original error: {e}"
    )

from embedding.create_embeddings import load_embedding_store, EmbeddingStore
from ingestion.ingest_documents import read_pdf, read_markdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuerySystem:
    
    def __init__(
        self,
        embedding_store: Optional[EmbeddingStore] = None,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        snippet_length: int = 500
    ):
        if embedding_store is None:
            logger.info("Loading embedding store from disk...")
            self.embedding_store = load_embedding_store(
                model_name=model_name,
                index_path=index_path,
                metadata_path=metadata_path
            )
        else:
            self.embedding_store = embedding_store
        
        self.model = self.embedding_store.model
        self.snippet_length = snippet_length
        logger.info(f"Query system initialized. Vector database contains {self.embedding_store.get_index_size()} documents.")
    
    def _read_document_content(self, filepath: str, file_type: str):
        try:
            if file_type == 'pdf':
                return read_pdf(filepath)
            elif file_type in ['markdown', 'md']:
                return read_markdown(filepath)
            else:
                logger.warning(f"Unknown file type: {file_type}. Attempting to read as text.")
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error reading document {filepath}: {e}")
            return ""
    
    def _extract_snippet(self, content: str, max_length: int = None):
        if max_length is None:
            max_length = self.snippet_length
        
        if len(content) <= max_length:
            return content
        
        # Truncate to max_length, trying to break at word boundaries
        snippet = content[:max_length]
        last_space = snippet.rfind(' ')
        if last_space > max_length * 0.8:  # Only use word boundary if it's not too early
            snippet = snippet[:last_space]
        
        return snippet + "..."
    
    def query_documents(
        self,
        query: str,
        top_k: int = 3
    ):
        if not query or not query.strip():
            logger.warning("Empty query received.")
            return []
        
        if self.embedding_store.index is None:
            raise ValueError("Vector index is not loaded. Please ensure embeddings are created.")
        
        if top_k <= 0:
            logger.warning(f"Invalid top_k value: {top_k}. Using default value of 3.")
            top_k = 3
        
        # Limit top_k to the number of documents in the index
        max_docs = self.embedding_store.get_index_size()
        if top_k > max_docs:
            logger.warning(f"Requested top_k={top_k} but only {max_docs} documents available. Using {max_docs}.")
            top_k = max_docs
        
        logger.info(f"Querying for: '{query}' (top_k={top_k})")
        
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        
        # Search the FAISS index
        distances, indices = self.embedding_store.index.search(query_embedding, top_k)
        
        # Process results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            # Skip invalid indices (FAISS can return -1 if there aren't enough documents)
            if idx < 0:
                continue
            
            # Get metadata
            metadata = self.embedding_store.get_metadata(idx)
            if metadata is None:
                logger.warning(f"No metadata found for index {idx}. Skipping.")
                continue
            
            # Read document content to get snippet
            filepath = metadata.get('filepath', '')
            file_type = metadata.get('file_type', '')
            
            snippet = ""
            if filepath:
                try:
                    content = self._read_document_content(filepath, file_type)
                    snippet = self._extract_snippet(content)
                except Exception as e:
                    logger.warning(f"Could not read content from {filepath}: {e}")
            
            result = {
                'index': int(idx),
                'title': metadata.get('title', ''),
                'filename': metadata.get('filename', ''),
                'filepath': filepath,
                'file_type': file_type,
                'snippet': snippet,
                'distance': float(distance)
            }
            results.append(result)
        
        logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
        return results


def query_documents(
    query: str,
    top_k: int = 3,
    query_system: Optional[QuerySystem] = None,
    **kwargs
):
    if query_system is None:
        query_system = QuerySystem(**kwargs)
    
    return query_system.query_documents(query, top_k)


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Knowledge Assistant - Query System")
    print("=" * 60)
    print()
    
    # Initialize query system
    try:
        print("Initializing query system...")
        qs = QuerySystem()
        print(f"✓ Loaded {qs.embedding_store.get_index_size()} documents")
        print()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create embeddings first by running: python embedding/create_embeddings.py")
        exit(1)
    except Exception as e:
        print(f"Error initializing query system: {e}")
        exit(1)
    
    # Interactive query loop
    print("Enter your queries below. Type 'exit', 'quit', or 'q' to exit.")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q', '']:
                print("\nExiting query system. Goodbye!")
                break
            
            # Query the system
            results = qs.query_documents(query, top_k=3)
            
            if not results:
                print("No results found.")
                continue
            
            # Display results
            print(f"\nFound {len(results)} relevant document(s):\n")
            for i, result in enumerate(results, 1):
                print(f"{'=' * 60}")
                print(f"Result {i} (Distance: {result['distance']:.4f})")
                print(f"{'=' * 60}")
                print(f"Title: {result['title']}")
                print(f"Filename: {result['filename']}")
                print(f"File Type: {result['file_type']}")
                print(f"\nSnippet:")
                print("-" * 60)
                print(result['snippet'])
                print("-" * 60)
                print()
        
        except KeyboardInterrupt:
            print("\n\nExiting query system. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")
            logger.exception("Error in query processing")
