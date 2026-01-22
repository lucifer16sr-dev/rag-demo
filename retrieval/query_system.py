
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import hashlib
from collections import OrderedDict

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
        model_name: str = "all-mpnet-base-v2",
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        snippet_length: int = 500,
        cache_size: int = 100,
        enable_cache: bool = True,
        relevance_threshold: float = 1.2
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
        
        # Initialize query cache (LRU-style using OrderedDict)
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.relevance_threshold = relevance_threshold
        self._cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Query system initialized. Vector database contains {self.embedding_store.get_index_size()} documents.")
        if enable_cache:
            logger.info(f"Query caching enabled (max size: {cache_size})")
        logger.info(f"Relevance threshold: {relevance_threshold} (results with distance > {relevance_threshold} will be filtered)")
    
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
    
    def _get_cache_key(self, query: str, top_k: int):
        # Normalize query (lowercase, strip whitespace) for consistent caching
        normalized_query = query.lower().strip()
        # Use hash to keep key size manageable
        query_hash = hashlib.md5(normalized_query.encode('utf-8')).hexdigest()[:16]
        return f"{query_hash}:{top_k}"
    
    def _get_from_cache(self, query: str, top_k: int):
        if not self.enable_cache:
            return None
        
        cache_key = self._get_cache_key(query, top_k)
        
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            self._cache_hits += 1
            logger.debug(f"Cache HIT for query: '{query[:50]}...' (top_k={top_k})")
            return self._cache[cache_key]
        
        self._cache_misses += 1
        logger.debug(f"Cache MISS for query: '{query[:50]}...' (top_k={top_k})")
        return None
    
    def _add_to_cache(self, query: str, top_k: int, results: List[Dict[str, Any]]):
        if not self.enable_cache:
            return
        
        cache_key = self._get_cache_key(query, top_k)
        
        # If cache is full, remove least recently used item
        if len(self._cache) >= self.cache_size and cache_key not in self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted entry for key: {oldest_key}")
        
        # Add new entry (or update existing)
        self._cache[cache_key] = results
        self._cache.move_to_end(cache_key)
        logger.debug(f"Cached results for query: '{query[:50]}...' (top_k={top_k})")
    
    def clear_cache(self):
        cache_size = len(self._cache)
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info(f"Cache cleared ({cache_size} entries removed)")
    
    def get_cache_stats(self):
        total_queries = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_queries * 100) if total_queries > 0 else 0.0
        
        return {
            'cache_enabled': self.enable_cache,
            'cache_size': len(self._cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_queries': total_queries,
            'hit_rate_percent': round(hit_rate, 2)
        }
    
    def query_documents(
        self,
        query: str,
        top_k: int = 5
    ):
        if not query or not query.strip():
            logger.warning("Empty query received.")
            return []
        
        if self.embedding_store.index is None:
            raise ValueError("Vector index is not loaded. Please ensure embeddings are created.")
        
        if top_k <= 0:
            logger.warning(f"Invalid top_k value: {top_k}. Using default value of 5.")
            top_k = 5
        
        # Limit top_k to the number of documents in the index
        max_docs = self.embedding_store.get_index_size()
        if top_k > max_docs:
            logger.warning(f"Requested top_k={top_k} but only {max_docs} documents available. Using {max_docs}.")
            top_k = max_docs
        
        # Check cache first
        cached_results = self._get_from_cache(query, top_k)
        if cached_results is not None:
            logger.info(f"Cache HIT - Returning cached results for query: '{query[:50]}...' (top_k={top_k})")
            logger.info(f"Retrieved {len(cached_results)} documents from cache")
            return cached_results
        
        # Cache miss - perform query
        logger.info(f"Querying for: '{query}' (top_k={top_k})")
        
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        
        # Search the FAISS index
        distances, indices = self.embedding_store.index.search(query_embedding, top_k)
        
        # Process results with relevance filtering
        results = []
        filtered_count = 0
        
        # Find the best (lowest) distance to use as a reference
        valid_distances = [d for d in distances[0] if d >= 0]
        if not valid_distances:
            logger.warning("No valid distances found in search results")
            return []
        
        best_distance = min(valid_distances)
        
        # If the best match is already very poor (> 1.8), likely an unrelated query
        # Filter everything in this case (increased from 1.5 to 1.8 to allow more legitimate queries)
        if best_distance > 1.8:
            logger.warning(f"Best distance {best_distance:.4f} is too high - likely unrelated query. Filtering all results.")
            return []
        
        # Use adaptive threshold for legitimate queries: 
        # - If best match is very good (< 0.5), use stricter filtering (best * 2.5)
        # - If best match is moderate (0.5-1.0), use moderate filtering (best * 2.0)
        # - If best match is borderline (1.0-1.5), use lenient filtering (best * 1.8)
        # - If best match is poor but acceptable (1.5-1.8), use very lenient filtering (best * 1.3)
        if best_distance < 0.5:
            adaptive_threshold = best_distance * 2.5
        elif best_distance < 1.0:
            adaptive_threshold = best_distance * 2.0
        elif best_distance < 1.5:
            adaptive_threshold = best_distance * 1.8
        else:  # 1.5 <= best_distance <= 1.8
            adaptive_threshold = best_distance * 1.3
        
        # Use the more lenient threshold (higher value = more lenient)
        # But cap it at 2.0 to prevent completely unrelated queries
        effective_threshold = min(max(adaptive_threshold, self.relevance_threshold), 2.0)
        
        logger.info(f"Best distance: {best_distance:.4f}, Adaptive threshold: {adaptive_threshold:.4f}, Absolute threshold: {self.relevance_threshold:.4f}, Effective threshold: {effective_threshold:.4f}")
        
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            # Skip invalid indices (FAISS can return -1 if there aren't enough documents)
            if idx < 0:
                continue
            
            # Apply effective threshold - only filter if distance exceeds the more lenient threshold
            # This ensures relevant queries like "Deployment" are not filtered out
            if distance > effective_threshold:
                filtered_count += 1
                logger.debug(f"Filtered result {idx} with distance {distance:.4f} (effective threshold: {effective_threshold:.4f})")
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
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} results above effective threshold ({effective_threshold:.4f})")
        
        if len(results) == 0 and filtered_count > 0:
            logger.warning(f"No relevant documents found for query: '{query}' (all {filtered_count} results filtered, best distance: {best_distance:.4f}, threshold: {effective_threshold:.4f})")
        else:
            logger.info(f"Retrieved {len(results)} relevant documents for query: '{query}' (best distance: {best_distance:.4f})")
        
        # Cache the results
        self._add_to_cache(query, top_k, results)
        
        # Log cache statistics periodically (every 10 queries)
        total_queries = self._cache_hits + self._cache_misses
        if total_queries % 10 == 0 and total_queries > 0:
            stats = self.get_cache_stats()
            logger.info(
                f"Cache stats - Hits: {stats['cache_hits']}, Misses: {stats['cache_misses']}, "
                f"Hit Rate: {stats['hit_rate_percent']}%, Size: {stats['cache_size']}/{stats['max_cache_size']}"
            )
        
        return results


def query_documents(
    query: str,
    top_k: int = 5,
    query_system: Optional[QuerySystem] = None,
    relevance_threshold: float = 0.8,
    **kwargs
):
    if query_system is None:
        kwargs.setdefault('relevance_threshold', relevance_threshold)
        query_system = QuerySystem(**kwargs)
    elif hasattr(query_system, 'relevance_threshold'):
        # Use existing threshold if query_system already has one
        pass
    
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
            results = qs.query_documents(query, top_k=5)
            
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
