
import sys
import time
import threading
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


def print_answer(result: dict):
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result.get('answer', 'No answer generated.'))
    print()
    
    sources = result.get('sources', [])
    if sources:
        print("=" * 60)
        print("SOURCES")
        print("=" * 60)
        for i, source in enumerate(sources, 1):
            print(f"{i}. {source}")
    else:
        print("No sources available.")
    
    print("=" * 60)
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
