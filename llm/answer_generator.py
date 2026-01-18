
import logging
import os
import re
from typing import List, Dict, Optional, Any

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from retrieval.query_system import query_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    
    def __init__(
        self,
        model_type: str = "auto",
        openai_model: str = "gpt-3.5-turbo",
        hf_model: str = "microsoft/DialoGPT-medium",
        use_local: bool = False
    ):
        self.model_type = model_type
        self.openai_model = openai_model
        self.hf_model = hf_model
        self.use_local = use_local
        
        # Initialize model based on availability and preferences
        self._initialize_model()
    
    def _initialize_model(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        has_openai_key = openai_key is not None and openai_key.strip() != ""
        
        # Determine which model to use
        if self.model_type == "auto":
            if self.use_local and TRANSFORMERS_AVAILABLE:
                model_type = "huggingface"
            elif has_openai_key and OPENAI_AVAILABLE and not self.use_local:
                model_type = "openai"
            elif TRANSFORMERS_AVAILABLE:
                model_type = "huggingface"
            else:
                model_type = "mock"
        else:
            model_type = self.model_type
        
        self.model_type = model_type
        self.llm = None
        
        if model_type == "openai":
            if not has_openai_key:
                logger.warning("OPENAI_API_KEY not found. Falling back to mock mode.")
                self.model_type = "mock"
                return
            
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI library not installed. Falling back to mock mode.")
                self.model_type = "mock"
                return
            
            # Set API key for old API pattern (<1.0) if needed
            try:
                # Try new API pattern first
                from openai import OpenAI
                # Just verify import works, we'll create client in _generate_with_openai
                logger.info(f"Initialized OpenAI model: {self.openai_model} (v1.0+ API)")
            except (ImportError, AttributeError):
                # Fall back to old API pattern
                openai.api_key = openai_key
                logger.info(f"Initialized OpenAI model: {self.openai_model} (legacy API)")
            
            self.llm = "openai"
        
        elif model_type == "huggingface":
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformers library not installed. Falling back to mock mode.")
                self.model_type = "mock"
                return
            
            try:
                logger.info(f"Loading Hugging Face model: {self.hf_model}...")
                # Use a simple text generation pipeline
                # Note: For better results, use a model fine-tuned for question answering
                self.llm = pipeline(
                    "text-generation",
                    model=self.hf_model,
                    tokenizer=self.hf_model,
                    max_length=512,
                    device=-1  # CPU by default
                )
                logger.info("Hugging Face model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading Hugging Face model: {e}")
                logger.warning("Falling back to mock mode.")
                self.model_type = "mock"
                self.llm = None
        
        else:  # mock
            logger.info("Using mock LLM mode (extracts first sentences from retrieved documents).")
            self.llm = "mock"
    
    def _extract_first_sentence(self, text: str):
        if not text:
            return ""
        
        # Remove leading whitespace
        text = text.strip()
        
        # Find sentence boundaries (., !, ? followed by space or end)
        sentence_end = re.search(r'[.!?]\s+', text)
        if sentence_end:
            return text[:sentence_end.end()].strip()
        
        # If no sentence boundary found, return first 200 chars or whole text
        return text[:200] if len(text) > 200 else text
    
    def _generate_with_openai(self, query: str, context: str):
        try:
            prompt = f"""Based on the following context, provide a concise answer to the question.

Context:
{context}

Question: {query}

Answer:"""
            
            # Try new API pattern first (OpenAI v1.0+)
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                answer = response.choices[0].message.content.strip()
            except (ImportError, AttributeError):
                # Fall back to old API pattern (<1.0)
                response = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                answer = response.choices[0].message.content.strip()
            
            return answer
        
        except Exception as e:
            logger.error(f"Error generating answer with OpenAI: {e}")
            raise
    
    def _generate_with_huggingface(self, query: str, context: str):
        try:
            # Create a prompt
            prompt = f"Question: {query}\n\nContext: {context}\n\nAnswer:"
            
            # Generate response
            result = self.llm(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm.tokenizer.eos_token_id
            )
            
            # Extract generated text (remove the prompt)
            generated_text = result[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            # Clean up the answer (remove extra newlines, etc.)
            answer = re.sub(r'\n+', ' ', answer).strip()
            
            # Limit answer length
            if len(answer) > 500:
                # Try to cut at sentence boundary
                sentence_end = re.search(r'[.!?]\s+', answer[:500])
                if sentence_end:
                    answer = answer[:sentence_end.end()].strip()
                else:
                    answer = answer[:500] + "..."
            
            return answer if answer else "I couldn't generate a coherent answer from the provided context."
        
        except Exception as e:
            logger.error(f"Error generating answer with Hugging Face: {e}")
            raise
    
    def _generate_with_mock(self, query: str, retrieved_docs: List[Dict[str, Any]]): # noqa: F821
        if not retrieved_docs:
            return "I couldn't find any relevant documents to answer your question."
        
        # Extract relevant content from each document snippet
        # Format it in a more readable way with better structure
        answer_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            snippet = doc.get('snippet', '')
            if snippet:
                # Clean up the snippet - remove excessive whitespace
                snippet = ' '.join(snippet.split())
                
                # Use a larger portion of the snippet (up to 250 chars per snippet)
                # Try to break at sentence boundary for cleaner cuts
                if len(snippet) > 250:
                    truncated = snippet[:250]
                    last_period = truncated.rfind('.')
                    last_question = truncated.rfind('?')
                    last_exclamation = truncated.rfind('!')
                    last_sentence = max(last_period, last_question, last_exclamation)
                    if last_sentence > 100:  # Only use if we have enough content
                        snippet_text = snippet[:last_sentence + 1]
                    else:
                        # If no good sentence break, cut at word boundary
                        last_space = truncated.rfind(' ')
                        if last_space > 100:
                            snippet_text = snippet[:last_space] + "..."
                        else:
                            snippet_text = snippet[:250] + "..."
                else:
                    snippet_text = snippet
                
                snippet_text = snippet_text.strip()
                if snippet_text:
                    answer_parts.append(snippet_text)
        
        if not answer_parts:
            return "I found relevant documents but couldn't extract meaningful information."
        
        # Combine snippets into an answer with better formatting
        # Use double newlines to create clear paragraph breaks
        if len(answer_parts) == 1:
            answer = answer_parts[0]
        else:
            # Format multiple snippets as separate paragraphs
            answer = "\n\n".join(f"• {part}" for part in answer_parts)
        
        # Limit total length
        if len(answer) > 1000:
            # Try to cut at a paragraph boundary
            truncated = answer[:1000]
            last_break = truncated.rfind('\n\n')
            if last_break > 500:
                answer = answer[:last_break]
            else:
                answer = truncated + "..."
        
        return answer
    
    def generate_answer(
        self,
        query: str,
        top_k: int = 3
    ): # noqa: F821
        if not query or not query.strip():
            logger.warning("Empty query received.")
            return {
                'answer': "Please provide a valid query.",
                'sources': [],
                'retrieved_text': ''
            }
        
        try:
            # Retrieve relevant documents
            logger.info(f"Retrieving top-{top_k} documents for query: '{query}'")
            retrieved_docs = query_documents(query, top_k=top_k)
            
            if not retrieved_docs:
                logger.warning(f"No documents retrieved for query: '{query}'")
                return {
                    'answer': "I couldn't find any relevant documents to answer your question.",
                    'sources': [],
                    'retrieved_text': ''
                }
            
            # Extract sources
            sources = []
            for doc in retrieved_docs:
                title = doc.get('title', '')
                filename = doc.get('filename', '')
                if title and filename:
                    sources.append(f"{title} ({filename})")
                elif title:
                    sources.append(title)
                elif filename:
                    sources.append(filename)
            
            # Combine retrieved text snippets
            retrieved_text = "\n\n".join([
                f"Document: {doc.get('title', doc.get('filename', 'Unknown'))}\n{doc.get('snippet', '')}"
                for doc in retrieved_docs
            ])
            
            # Generate answer based on model type
            logger.info(f"Generating answer using {self.model_type} model...")
            
            if self.model_type == "openai":
                answer = self._generate_with_openai(query, retrieved_text)
            
            elif self.model_type == "huggingface":
                answer = self._generate_with_huggingface(query, retrieved_text)
            
            else:  # mock
                answer = self._generate_with_mock(query, retrieved_docs)
            
            logger.info(f"Successfully generated answer (length: {len(answer)} chars)")
            
            return {
                'answer': answer,
                'sources': sources,
                'retrieved_text': retrieved_text
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            logger.exception("Full error details:")
            return {
                'answer': f"An error occurred while generating the answer: {str(e)}",
                'sources': [],
                'retrieved_text': ''
            }


def generate_answer(
    query: str,
    top_k: int = 3,
    answer_generator: Optional[AnswerGenerator] = None,
    **kwargs
): # noqa: F821
    if answer_generator is None:
        answer_generator = AnswerGenerator(**kwargs)
    
    return answer_generator.generate_answer(query, top_k)


if __name__ == "__main__":          
    print("=" * 60)
    print("RAG Knowledge Assistant - Answer Generator")
    print("=" * 60)
    print()
    
    # Check for API keys and model availability
    openai_key = os.getenv("OPENAI_API_KEY")
    print(f"OpenAI API Key: {'Available' if openai_key else 'Not found'}")
    print(f"Transformers: {'Available' if TRANSFORMERS_AVAILABLE else 'Not installed'}")
    print()
    
    # Initialize answer generator
    try:
        print("Initializing answer generator...")
        ag = AnswerGenerator()
        print(f"✓ Using model type: {ag.model_type}")
        print()
    except Exception as e:
        print(f"Error initializing answer generator: {e}")
        exit(1)
    
    # Interactive query loop
    print("Enter your queries below. Type 'exit', 'quit', or 'q' to exit.")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q', '']:
                print("\nExiting answer generator. Goodbye!")
                break
            
            # Generate answer
            result = ag.generate_answer(query, top_k=3)
            
            # Display results
            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(result['answer'])
            print()
            
            if result['sources']:
                print("=" * 60)
                print("SOURCES")
                print("=" * 60)
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source}")
                print()
            
            print("-" * 60)
        
        except KeyboardInterrupt:
            print("\n\nExiting answer generator. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")
            logger.exception("Error in answer generation")
