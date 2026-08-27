"""
RAG Engine for Loan Defaulter Recovery Recommendations.

Sync API:
    from src.rag import ingest_policy, get_recommendation

Async API (for FastAPI/async backends):
    from src.rag import aget_recommendation

Integration utilities:
    from src.rag import ColumnMapper, parse_recommendation, format_for_api
"""

from .mapper import ColumnMapper
from .pdf_loader import PDFLoadError
from .rag_engine import (
    aget_recommendation,
    clear_cache,
    get_recommendation,
    get_store_info,
    ingest_all_policies,
    ingest_policy,
    replace_policy,
)
from .response_parser import format_for_api, parse_recommendation
from .sanitize import sanitize_customer_data

__all__ = [
    # Ingestion
    "ingest_policy",
    "ingest_all_policies",
    "replace_policy",
    # Recommendation
    "get_recommendation",
    "aget_recommendation",
    # Info & cache
    "get_store_info",
    "clear_cache",
    # Integration helpers
    "ColumnMapper",
    "parse_recommendation",
    "format_for_api",
    # Sanitization
    "sanitize_customer_data",
    # Errors
    "PDFLoadError",
]
