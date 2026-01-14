"""
Ingestion package for document processing.
"""

from .ingest_documents import (
    ingest_all_documents,
    ingest_pdf_files,
    ingest_markdown_files,
    read_pdf,
    read_markdown,
    clean_text,
    get_document_by_title
)

__all__ = [
    'ingest_all_documents',
    'ingest_pdf_files',
    'ingest_markdown_files',
    'read_pdf',
    'read_markdown',
    'clean_text',
    'get_document_by_title'
]

