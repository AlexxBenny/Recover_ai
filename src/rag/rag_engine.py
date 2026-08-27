"""
Core RAG engine for loan defaulter recovery recommendations.

Public API (sync):
    ingest_policy(pdf_path)        -- Parse a policy PDF and add to vector store
    ingest_all_policies(dir)       -- Ingest all PDFs from a directory
    replace_policy(pdf_path)       -- Replace an existing policy
    get_recommendation(customer)   -- Get collection strategy recommendation
    get_store_info()               -- Get vector store status info

Public API (async):
    aget_recommendation(customer)  -- Async version of get_recommendation

Utilities:
    clear_cache()                  -- Clear the response cache
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

from . import config
from .cache import ResponseCache
from .pdf_loader import PDFLoadError, extract_text_from_pdf
from .prompts import SYSTEM_PROMPT, build_retrieval_query, format_customer_query
from .sanitize import sanitize_customer_data
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialized)
# ---------------------------------------------------------------------------
_vectorstore: Optional[VectorStore] = None
_cache: Optional[ResponseCache] = None


def _get_vectorstore() -> VectorStore:
    """Get or create the vector store singleton."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore()
    return _vectorstore


def _get_cache() -> ResponseCache:
    """Get or create the response cache singleton."""
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache


# ---------------------------------------------------------------------------
# Public API -- Ingestion
# ---------------------------------------------------------------------------


def ingest_policy(pdf_path: Union[str, Path]) -> dict:
    """
    Ingest a single policy PDF into the vector store.

    This is the function the backend teammate will call when the admin
    uploads a new policy document.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dict with ingestion results::

            {
                "success": True,
                "source": "policy_name.pdf",
                "pages": 3,
                "chunks_added": 12,
                "error": None
            }
    """
    pdf_path = Path(pdf_path)

    try:
        # Extract text from PDF (handles validation internally)
        documents = extract_text_from_pdf(pdf_path)

        if not documents:
            return {
                "success": False,
                "source": pdf_path.name,
                "pages": 0,
                "chunks_added": 0,
                "error": "No extractable text found in PDF.",
            }

        # Add to vector store
        vs = _get_vectorstore()
        chunks_added = vs.add_documents(documents)

        logger.info(
            f"Ingested '{pdf_path.name}': {len(documents)} pages, "
            f"{chunks_added} chunks"
        )

        return {
            "success": True,
            "source": pdf_path.name,
            "pages": len(documents),
            "chunks_added": chunks_added,
            "error": None,
        }

    except (FileNotFoundError, ValueError, PDFLoadError) as e:
        logger.error(f"Failed to ingest '{pdf_path.name}': {e}")
        return {
            "success": False,
            "source": pdf_path.name,
            "pages": 0,
            "chunks_added": 0,
            "error": str(e),
        }
    except Exception as e:
        logger.exception(f"Unexpected error ingesting '{pdf_path.name}'")
        return {
            "success": False,
            "source": pdf_path.name,
            "pages": 0,
            "chunks_added": 0,
            "error": f"Unexpected error: {e}",
        }


def ingest_all_policies(
    policies_dir: Optional[Union[str, Path]] = None,
) -> list[dict]:
    """
    Ingest all PDF files from a directory.

    Args:
        policies_dir: Directory containing policy PDFs.
            Defaults to config.POLICIES_DIR ("./policies/")

    Returns:
        List of ingestion result dicts (one per PDF).
        Each dict includes a "success" boolean and "error" message if failed.

    Raises:
        FileNotFoundError: If the policies directory doesn't exist
    """
    policies_dir = Path(policies_dir or config.POLICIES_DIR)

    if not policies_dir.exists():
        raise FileNotFoundError(f"Policies directory not found: {policies_dir}")

    results = []
    for pdf_file in sorted(policies_dir.glob("*.pdf")):
        result = ingest_policy(pdf_file)
        results.append(result)

    return results


def replace_policy(pdf_path: Union[str, Path]) -> dict:
    """
    Replace an existing policy -- deletes old chunks, ingests new PDF.

    Use this when the admin uploads an updated version of a policy.

    Args:
        pdf_path: Path to the new PDF file

    Returns:
        Dict with: {success, source, old_chunks_deleted, new_chunks_added, error}
    """
    pdf_path = Path(pdf_path)

    try:
        vs = _get_vectorstore()

        # Delete old chunks for this policy
        old_deleted = vs.delete_policy(pdf_path.name)

        # Ingest the new version
        result = ingest_policy(pdf_path)

        # Clear cache since policies changed
        if config.CACHE_ENABLED:
            _get_cache().clear()
            logger.info("Cache cleared after policy replacement")

        return {
            "success": result["success"],
            "source": pdf_path.name,
            "old_chunks_deleted": old_deleted,
            "new_chunks_added": result["chunks_added"],
            "error": result.get("error"),
        }

    except Exception as e:
        logger.exception(f"Failed to replace policy '{pdf_path.name}'")
        return {
            "success": False,
            "source": pdf_path.name,
            "old_chunks_deleted": 0,
            "new_chunks_added": 0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Public API -- Recommendation (sync)
# ---------------------------------------------------------------------------


def get_recommendation(customer: dict) -> dict:
    """
    Get a collection strategy recommendation for a defaulter.

    Pipeline:
    1. Sanitize customer data (strip PII)
    2. Check response cache
    3. Build retrieval query from customer profile
    4. Retrieve relevant policy chunks from vector store
    5. Format prompt with customer details + policy context
    6. Call LLM for recommendation
    7. Cache and return result

    Args:
        customer: Dict with customer details using Lending Club column names.
            Key fields include:
            - loan_amnt (int): Loan amount
            - loan_status (str): 'In Grace Period', 'Late (31-120 days)',
              'Default', 'Charged Off', etc.
            - grade (str): Loan risk grade (A-G)
            - sub_grade (str): Sub-grade (e.g., 'A4', 'C5')
            - purpose, annual_inc, dti, home_ownership, term, int_rate, etc.

    Returns:
        Dict with::

            {
                "recommendation": "Full LLM response text",
                "retrieved_policies": [...],
                "customer_summary": {...},
                "pii_fields_removed": [...],
                "cached": False
            }
    """
    # Step 1: Sanitize -- remove any PII fields
    sanitized, pii_removed = sanitize_customer_data(customer)

    # Step 2: Check cache
    if config.CACHE_ENABLED:
        cached_result = _get_cache().get(sanitized)
        if cached_result is not None:
            cached_result["cached"] = True
            cached_result["pii_fields_removed"] = pii_removed
            return cached_result

    # Step 3-6: Core RAG pipeline
    result = _run_rag_pipeline(sanitized)
    result["pii_fields_removed"] = pii_removed
    result["cached"] = False

    # Step 7: Cache the result
    if config.CACHE_ENABLED:
        _get_cache().set(sanitized, result)

    return result


# ---------------------------------------------------------------------------
# Public API -- Recommendation (async)
# ---------------------------------------------------------------------------


async def aget_recommendation(customer: dict) -> dict:
    """
    Async version of get_recommendation.

    Same pipeline as get_recommendation but uses async LLM calls.
    Use this in async web frameworks (FastAPI, aiohttp, etc.).

    Args:
        customer: Same as get_recommendation

    Returns:
        Same as get_recommendation
    """
    # Step 1: Sanitize
    sanitized, pii_removed = sanitize_customer_data(customer)

    # Step 2: Check cache
    if config.CACHE_ENABLED:
        cached_result = _get_cache().get(sanitized)
        if cached_result is not None:
            cached_result["cached"] = True
            cached_result["pii_fields_removed"] = pii_removed
            return cached_result

    # Step 3-4: Retrieval (sync -- fast enough, no async needed)
    vs = _get_vectorstore()
    retrieval_query = build_retrieval_query(sanitized)
    retrieved = vs.query(retrieval_query)

    if not retrieved:
        return {
            "recommendation": (
                "No policies found in the knowledge base. "
                "Please ingest policy documents first."
            ),
            "retrieved_policies": [],
            "customer_summary": sanitized,
            "pii_fields_removed": pii_removed,
            "cached": False,
        }

    # Step 5: Build prompt
    context = "\n\n---\n\n".join(
        f"[Source: {r['metadata']['source']}, Page {r['metadata']['page_number']}]\n{r['text']}"
        for r in retrieved
    )
    user_prompt = format_customer_query(sanitized, context)

    # Step 6: Async LLM call
    llm_response = await _acall_llm(user_prompt)

    result = {
        "recommendation": llm_response,
        "retrieved_policies": [
            {
                "source": r["metadata"]["source"],
                "page": r["metadata"]["page_number"],
                "relevance_score": r["score"],
                "excerpt": (
                    r["text"][:200] + "..."
                    if len(r["text"]) > 200
                    else r["text"]
                ),
            }
            for r in retrieved
        ],
        "customer_summary": sanitized,
        "pii_fields_removed": pii_removed,
        "cached": False,
    }

    # Step 7: Cache
    if config.CACHE_ENABLED:
        _get_cache().set(sanitized, result)

    return result


# ---------------------------------------------------------------------------
# Public API -- Store Info & Cache
# ---------------------------------------------------------------------------


def get_store_info() -> dict:
    """Get information about the current vector store state."""
    vs = _get_vectorstore()
    info = {
        "total_chunks": vs.count,
        "policies_loaded": vs.list_policies(),
        "persist_dir": vs.persist_dir,
        "embedding_model": vs.embedding_model_name,
        "collection_name": vs.collection_name,
    }

    if config.CACHE_ENABLED:
        info["cache"] = _get_cache().stats

    return info


def clear_cache() -> None:
    """Clear the response cache."""
    if config.CACHE_ENABLED:
        _get_cache().clear()


# ---------------------------------------------------------------------------
# Internal -- RAG Pipeline (shared by sync and async)
# ---------------------------------------------------------------------------


def _run_rag_pipeline(customer: dict) -> dict:
    """Core RAG pipeline: retrieve -> prompt -> LLM -> result."""
    vs = _get_vectorstore()

    # Build retrieval query
    retrieval_query = build_retrieval_query(customer)

    # Retrieve relevant policy chunks
    retrieved = vs.query(retrieval_query)

    if not retrieved:
        return {
            "recommendation": (
                "No policies found in the knowledge base. "
                "Please ingest policy documents first using "
                "ingest_policy() or ingest_all_policies()."
            ),
            "retrieved_policies": [],
            "customer_summary": customer,
        }

    # Build context from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[Source: {r['metadata']['source']}, Page {r['metadata']['page_number']}]\n{r['text']}"
        for r in retrieved
    )

    # Build the full prompt
    user_prompt = format_customer_query(customer, context)

    # Call the LLM
    llm_response = _call_llm(user_prompt)

    return {
        "recommendation": llm_response,
        "retrieved_policies": [
            {
                "source": r["metadata"]["source"],
                "page": r["metadata"]["page_number"],
                "relevance_score": r["score"],
                "excerpt": (
                    r["text"][:200] + "..."
                    if len(r["text"]) > 200
                    else r["text"]
                ),
            }
            for r in retrieved
        ],
        "customer_summary": customer,
    }


# ---------------------------------------------------------------------------
# LLM Calls -- Sync
# ---------------------------------------------------------------------------


def _call_llm(prompt: str) -> str:
    """Route to the configured LLM provider (sync)."""
    provider = config.LLM_PROVIDER.lower()

    if provider == "google":
        return _call_google(prompt)
    elif provider == "openai":
        return _call_openai(prompt)
    elif provider == "ollama":
        return _call_ollama(prompt)
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            "Set LLM_PROVIDER to 'google', 'openai', or 'ollama'."
        )


def _call_google(prompt: str) -> str:
    """Call Google Gemini via google-genai SDK."""
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not set. Add it to your .env file. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=config.LLM_TEMPERATURE,
        ),
    )

    return response.text


def _call_openai(prompt: str) -> str:
    """Call OpenAI-compatible API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=config.LLM_TEMPERATURE,
    )

    return response.choices[0].message.content


def _call_ollama(prompt: str) -> str:
    """Call Ollama (local LLM server)."""
    import requests

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": config.LLM_MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.LLM_TEMPERATURE,
            },
        },
    )
    response.raise_for_status()
    return response.json()["response"]


# ---------------------------------------------------------------------------
# LLM Calls -- Async
# ---------------------------------------------------------------------------


async def _acall_llm(prompt: str) -> str:
    """Route to the configured LLM provider (async)."""
    provider = config.LLM_PROVIDER.lower()

    if provider == "google":
        return await _acall_google(prompt)
    elif provider == "openai":
        return await _acall_openai(prompt)
    elif provider == "ollama":
        return await _acall_ollama(prompt)
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            "Set LLM_PROVIDER to 'google', 'openai', or 'ollama'."
        )


async def _acall_google(prompt: str) -> str:
    """Async call to Google Gemini."""
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not set. Add it to your .env file. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)

    response = await client.aio.models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=config.LLM_TEMPERATURE,
        ),
    )

    return response.text


async def _acall_openai(prompt: str) -> str:
    """Async call to OpenAI."""
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")

    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=config.LLM_TEMPERATURE,
    )

    return response.choices[0].message.content


async def _acall_ollama(prompt: str) -> str:
    """Async call to Ollama using httpx."""
    import httpx

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{ollama_url}/api/generate",
            json={
                "model": config.LLM_MODEL,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                },
            },
        )
        response.raise_for_status()
        return response.json()["response"]
