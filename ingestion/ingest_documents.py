
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import re

try:
    import PyPDF2
except ImportError:
    raise ImportError("PyPDF2 is required. Install with: pip install PyPDF2")

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available. Language detection disabled. Install with: pip install langdetect")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def detect_language(text: str):
    if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 50:
        return None
    
    try:
        # Use first 1000 characters for faster detection
        sample_text = text[:1000] if len(text) > 1000 else text
        language = detect(sample_text)
        return language
    except LangDetectException as e:
        logger.debug(f"Language detection failed: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error in language detection: {e}")
        return None


def clean_text(text: str, remove_extra_spaces: bool = True):
    if not text:
        return ""
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize unicode characters (optional - can be enabled if needed)
    # text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace between words (but keep single spaces)
    if remove_extra_spaces:
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
    if remove_extra_spaces:
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
        logger.info(f"No PDF files found in {data_dir}")
        return documents
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) in {data_dir}")
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing PDF: {pdf_file.name}")
            raw_text = read_pdf(str(pdf_file))
            
            if not raw_text or not raw_text.strip():
                logger.warning(f"⚠ Skipping {pdf_file.name}: No text extracted")
                continue
            
            # Clean text
            cleaned_text = clean_text(raw_text)
            
            if not cleaned_text:
                logger.warning(f"⚠ Skipping {pdf_file.name}: Text became empty after cleaning")
                continue
            
            # Detect language
            detected_language = detect_language(cleaned_text)
            
            # Create document entry
            doc = {
                'title': pdf_file.stem,  # Filename without extension
                'filename': pdf_file.name,
                'filepath': str(pdf_file),
                'content': cleaned_text,
                'file_type': 'pdf',
                'language': detected_language or 'unknown'
            }
            
            if detected_language:
                logger.info(f"✓ Processed {pdf_file.name} (language: {detected_language}, {len(cleaned_text)} chars)")
            else:
                logger.info(f"✓ Processed {pdf_file.name} ({len(cleaned_text)} chars)")
            
            documents.append(doc)
        
        except FileNotFoundError as e:
            logger.warning(f"⚠ Skipping {pdf_file.name}: {e}")
            continue
        except Exception as e:
            logger.warning(f"⚠ Skipping {pdf_file.name}: Error processing file - {e}")
            continue
    
    logger.info(f"Successfully processed {len(documents)}/{len(pdf_files)} PDF file(s)")
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
        logger.info(f"No Markdown files found in {data_dir}")
        return documents
    
    logger.info(f"Found {len(markdown_files)} Markdown file(s) in {data_dir}")
    
    for md_file in markdown_files:
        try:
            logger.info(f"Processing Markdown: {md_file.name}")
            raw_text = read_markdown(str(md_file))
            
            if not raw_text or not raw_text.strip():
                logger.warning(f"⚠ Skipping {md_file.name}: No text extracted")
                continue
            
            # Clean text
            cleaned_text = clean_text(raw_text)
            
            if not cleaned_text:
                logger.warning(f"⚠ Skipping {md_file.name}: Text became empty after cleaning")
                continue
            
            # Detect language
            detected_language = detect_language(cleaned_text)
            
            # Create document entry
            doc = {
                'title': md_file.stem,  # Filename without extension
                'filename': md_file.name,
                'filepath': str(md_file),
                'content': cleaned_text,
                'file_type': 'markdown',
                'language': detected_language or 'unknown'
            }
            
            if detected_language:
                logger.info(f"✓ Processed {md_file.name} (language: {detected_language}, {len(cleaned_text)} chars)")
            else:
                logger.info(f"✓ Processed {md_file.name} ({len(cleaned_text)} chars)")
            
            documents.append(doc)
        
        except FileNotFoundError as e:
            logger.warning(f"⚠ Skipping {md_file.name}: {e}")
            continue
        except Exception as e:
            logger.warning(f"⚠ Skipping {md_file.name}: Error processing file - {e}")
            continue
    
    logger.info(f"Successfully processed {len(documents)}/{len(markdown_files)} Markdown file(s)")
    return documents


def ingest_all_documents(data_dir: str = "data"):
    logger.info(f"Starting document ingestion from: {data_dir}")
    all_documents = []
    
    # Ingest PDF files (continues on error, logs warnings)
    try:
        pdf_documents = ingest_pdf_files(data_dir)
        all_documents.extend(pdf_documents)
    except Exception as e:
        logger.error(f"Error ingesting PDF files: {e}")
    
    # Ingest Markdown files (continues on error, logs warnings)
    try:
        md_documents = ingest_markdown_files(data_dir)
        all_documents.extend(md_documents)
    except Exception as e:
        logger.error(f"Error ingesting Markdown files: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Document ingestion completed:")
    logger.info(f"  Total documents: {len(all_documents)}")
    logger.info(f"  - PDF files: {len([d for d in all_documents if d.get('file_type') == 'pdf'])}")
    logger.info(f"  - Markdown files: {len([d for d in all_documents if d.get('file_type') == 'markdown'])}")
    logger.info(f"{'='*60}\n")
    
    return all_documents


def get_document_by_title(documents: List[Dict[str, str]], title: str):
    for doc in documents:
        if doc['title'] == title:
            return doc
    return None

