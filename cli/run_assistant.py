
import sys
import time
import threading
import re
from typing import Optional

from llm.answer_generator import generate_answer


class LoadingIndicator:
    
    def __init__(self, message: str = "Generating answer"):
        self.message = message
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def _animate(self):     
        chars = "|/-\\"
        i = 0
        while self.running:
            sys.stdout.write(f"\r{self.message} {chars[i % len(chars)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        sys.stdout.flush()


def highlight_keywords_cli(text: str, keywords: list):
    if not keywords or not text:
        return text
    
    result = text
    # Sort by length (longest first) to avoid partial matches
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    for keyword in sorted_keywords:
        # Use ANSI escape codes for yellow background
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        result = pattern.sub(
            lambda m: f"\033[43m{m.group()}\033[0m",
            result
        )
    
    return result


def print_answer(result: dict):
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    
    answer = result.get('answer', 'No answer generated.')
    keywords = result.get('keywords', [])
    no_relevant = result.get("no_relevant_results", False)
    
    # Show warning if no relevant results
    if no_relevant:
        print("\033[93m⚠️  WARNING: No relevant documents found for this query.\033[0m")
        print()
    
    # Format answer with paragraphs and highlight keywords
    if answer:
        # Split into paragraphs for better display
        paragraphs = answer.split('\n\n')
        for para in paragraphs:
            if para.strip():
                highlighted_para = highlight_keywords_cli(para, keywords)
                print(highlighted_para)
                print()  # Blank line between paragraphs
    else:
        print(answer)
    
    print()
    
    # Display sources with detailed information
    sources_detailed = result.get("sources_detailed", [])
    sources = result.get("sources", [])
    
    if sources_detailed or sources:
        print("=" * 60)
        print("SOURCES")
        print("=" * 60)
        
        if sources_detailed:
            for i, source_info in enumerate(sources_detailed, 1):
                filename = source_info.get('filename', 'Unknown')
                filepath = source_info.get('filepath', '')
                title = source_info.get('title', filename)
                file_type = source_info.get('file_type', '')
                distance = source_info.get('distance', 0.0)
                
                print(f"{i}. {title} ({filename})")
                print(f"   Type: {file_type} | Relevance: {distance:.4f}")
                if filepath:
                    print(f"   📄 {filepath}")
                print()
        else:
            # Fallback to simple sources
            for i, source in enumerate(sources, 1):
                print(f"{i}. {source}")
        
        print("=" * 60)
    else:
        print("No sources available.")
        print("=" * 60)
    
    # Display keywords
    if keywords:
        print(f"\nKeywords: {', '.join(keywords[:10])}")
    
    print()


def main():
    print("=" * 60)
    print("Welcome to RAG Knowledge Assistant!")
    print("=" * 60)
    print()
    print("Type your questions below. Type 'exit', 'quit', or 'q' to exit.")
    print("-" * 60)
    print()
    
    # Interactive query loop
    while True:
        try:
            # Get user input
            query = input("Query: ").strip()
            
            # Handle exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nThank you for using RAG Knowledge Assistant. Goodbye!")
                break
            
            # Handle empty input
            if not query:
                print("Please enter a valid query or type 'exit' to quit.\n")
                continue
            
            # Show loading indicator
            indicator = LoadingIndicator("Generating answer")
            indicator.start()
            
            try:
                # Generate answer
                result = generate_answer(query, top_k=5)
            except KeyboardInterrupt:
                indicator.stop()
                print("\n\nOperation cancelled by user.")
                continue
            except Exception as e:
                indicator.stop()
                print(f"\n\nError generating answer: {str(e)}")
                print("Please try again or type 'exit' to quit.\n")
                continue
            finally:
                indicator.stop()
            
            # Print results
            print_answer(result)
        
        except KeyboardInterrupt:
            print("\n\nThank you for using RAG Knowledge Assistant. Goodbye!")
            break
        except EOFError:
            # Handle Ctrl+D (EOF)
            print("\n\nThank you for using RAG Knowledge Assistant. Goodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("Please try again or type 'exit' to quit.\n")


if __name__ == "__main__":
    main()
