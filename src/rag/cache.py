"""
Simple in-memory response cache for RAG recommendations.

Avoids redundant LLM calls for the same customer profile.
Uses a hash of the customer dict as the cache key with TTL-based expiry.
"""

import hashlib
import json
import logging
import time
from typing import Optional

from . import config

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    In-memory LRU cache with TTL for recommendation responses.

    Usage:
        cache = ResponseCache(max_size=100, ttl_seconds=3600)

        # Check cache before calling LLM
        cached = cache.get(customer_dict)
        if cached:
            return cached

        # After LLM call, store result
        cache.set(customer_dict, result)
    """

    def __init__(
        self,
        max_size: int = None,
        ttl_seconds: int = None,
    ):
        self._cache: dict[str, dict] = {}
        self._max_size = max_size or config.CACHE_MAX_SIZE
        self._ttl = ttl_seconds or config.CACHE_TTL_SECONDS

    def _make_key(self, customer: dict) -> str:
        """Create a deterministic hash key from customer dict."""
        # Sort keys for consistency, convert values to strings
        serialized = json.dumps(customer, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def get(self, customer: dict) -> Optional[dict]:
        """
        Retrieve cached result for a customer profile.

        Returns None on cache miss or if the entry has expired.
        """
        key = self._make_key(customer)

        if key not in self._cache:
            return None

        entry = self._cache[key]
        age = time.time() - entry["timestamp"]

        if age > self._ttl:
            # Expired — remove and return miss
            del self._cache[key]
            logger.debug(f"Cache expired for key {key} (age: {age:.0f}s)")
            return None

        logger.info(f"Cache hit for key {key} (age: {age:.0f}s)")
        return entry["result"]

    def set(self, customer: dict, result: dict) -> None:
        """Store a result in the cache."""
        # Evict oldest entry if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(
                self._cache, key=lambda k: self._cache[k]["timestamp"]
            )
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted oldest entry (size: {self._max_size})")

        key = self._make_key(customer)
        self._cache[key] = {
            "result": result,
            "timestamp": time.time(),
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Cache cleared")

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._cache)

    @property
    def stats(self) -> dict:
        """Cache statistics."""
        now = time.time()
        ages = [now - e["timestamp"] for e in self._cache.values()]
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "oldest_entry_age": max(ages) if ages else 0,
            "newest_entry_age": min(ages) if ages else 0,
        }
