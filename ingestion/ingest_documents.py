
import os
from pathlib import Path
from typing import List, Dict, Optional
import re

try:
    import PyPDF2
except ImportError:
    raise ImportError("PyPDF2 is required. Install with: pip install PyPDF2")


def read_pdf(file_path: str):
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    if file_path.suffix.lower() != '.pdf':
        raise ValueError(f"File is not a PDF: {file_path}")
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text_content.append(page_text)
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1} of {file_path}: {e}")
            
            return '\n'.join(text_content)
    
    except Exception as e:
        raise ValueError(f"Error reading PDF file {file_path}: {e}")


def read_markdown(file_path: str):
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")
    
    if file_path.suffix.lower() not in ['.md', '.markdown']:
        raise ValueError(f"File is not a Markdown file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
                return content
        except Exception as e:
            raise ValueError(f"Error reading Markdown file {file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading Markdown file {file_path}: {e}")


def clean_text(text: str):
    if not text:
        return ""
    
    # Remove extra whitespace between words (but keep single spaces)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize line breaks (convert all to \n)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    # Remove multiple consecutive line breaks (keep max 2 for paragraph separation)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    
    # Remove empty lines at the start and end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    
    # Join lines back together
    cleaned_text = '\n'.join(lines)
    
    # Final cleanup: remove any remaining excessive whitespace
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
    
    return cleaned_text.strip()


def ingest_pdf_files(data_dir: str = "data"):
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not data_path.is_dir():
        raise ValueError(f"Path is not a directory: {data_dir}")
    
    documents = []
    pdf_files = list(data_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return documents
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing PDF: {pdf_file.name}")
            raw_text = read_pdf(str(pdf_file))
            cleaned_text = clean_text(raw_text)
            
            if cleaned_text:  # Only add if there's actual content
                documents.append({
                    'title': pdf_file.stem,  # Filename without extension
                    'filename': pdf_file.name,
                    'filepath': str(pdf_file),
                    'content': cleaned_text,
                    'file_type': 'pdf'
                })
            else:
                print(f"Warning: No text extracted from {pdf_file.name}")
        
        except Exception as e:
            print(f"Error processing PDF {pdf_file.name}: {e}")
    
    return documents


def ingest_markdown_files(data_dir: str = "data"):
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not data_path.is_dir():
        raise ValueError(f"Path is not a directory: {data_dir}")
    
    documents = []
    markdown_files = list(data_path.glob("*.md")) + list(data_path.glob("*.markdown"))
    
    if not markdown_files:
        print(f"No Markdown files found in {data_dir}")
        return documents
    
    for md_file in markdown_files:
        try:
            print(f"Processing Markdown: {md_file.name}")
            raw_text = read_markdown(str(md_file))
            cleaned_text = clean_text(raw_text)
            
            if cleaned_text:  # Only add if there's actual content
                documents.append({
                    'title': md_file.stem,  # Filename without extension
                    'filename': md_file.name,
                    'filepath': str(md_file),
                    'content': cleaned_text,
                    'file_type': 'markdown'
                })
            else:
                print(f"Warning: No text extracted from {md_file.name}")
        
        except Exception as e:
            print(f"Error processing Markdown {md_file.name}: {e}")
    
    return documents


def ingest_all_documents(data_dir: str = "data") -> List[Dict[str, str]]:
    all_documents = []
    
    # Ingest PDF files
    pdf_documents = ingest_pdf_files(data_dir)
    all_documents.extend(pdf_documents)
    
    # Ingest Markdown files
    md_documents = ingest_markdown_files(data_dir)
    all_documents.extend(md_documents)
    
    print(f"\nTotal documents ingested: {len(all_documents)}")
    print(f"  - PDF files: {len(pdf_documents)}")
    print(f"  - Markdown files: {len(md_documents)}")
    
    return all_documents


def get_document_by_title(documents: List[Dict[str, str]], title: str):
    for doc in documents:
        if doc['title'] == title:
            return doc
    return None

