"""
Response parser for structuring LLM recommendation output.

Parses the raw markdown recommendation into a structured dict
that frontends can render directly (e.g., risk badge, strategy cards,
settlement terms panel).

Usage:
    from src.rag.response_parser import parse_recommendation

    result = get_recommendation(customer)
    structured = parse_recommendation(result["recommendation"])

    # Now you have:
    structured["customer_summary"]    # str
    structured["dpd_bucket"]          # str
    structured["risk_level"]          # str - "Low" / "Medium" / "High" / "Critical"
    structured["strategy"]            # str
    structured["channels"]            # str
    structured["escalation"]          # str
    structured["settlement"]          # str
    structured["reasoning"]           # str
    structured["regulatory_notes"]    # str
"""

import re
from typing import Optional


# Section header patterns (matching the 9-part structure in prompts.py)
SECTION_PATTERNS = [
    ("customer_summary", r"(?:1\.\s*\*?\*?Customer Status Summary\*?\*?|###\s*1\..*Summary)"),
    ("dpd_bucket", r"(?:2\.\s*\*?\*?DPD Bucket\*?\*?|###\s*2\..*Bucket)"),
    ("risk_level", r"(?:3\.\s*\*?\*?Risk Level\*?\*?|###\s*3\..*Risk)"),
    ("strategy", r"(?:4\.\s*\*?\*?Recommended Strategy\*?\*?|###\s*4\..*Strateg)"),
    ("channels", r"(?:5\.\s*\*?\*?Communication Channel\*?\*?|###\s*5\..*Channel)"),
    ("escalation", r"(?:6\.\s*\*?\*?Escalation\*?\*?|###\s*6\..*Escalat)"),
    ("settlement", r"(?:7\.\s*\*?\*?Settlement\*?\*?|###\s*7\..*Settlement)"),
    ("reasoning", r"(?:8\.\s*\*?\*?Reasoning\*?\*?|###\s*8\..*Reason)"),
    ("regulatory_notes", r"(?:9\.\s*\*?\*?Regulatory\*?\*?|###\s*9\..*Regulat)"),
]


def parse_recommendation(text: str) -> dict:
    """
    Parse a raw LLM recommendation into structured sections.

    Args:
        text: Raw markdown recommendation from the LLM

    Returns:
        Dict with parsed sections. Each value is the raw text content
        of that section. Sections not found will have None values.
    """
    if not text:
        return {name: None for name, _ in SECTION_PATTERNS}

    result = {}

    # Find section boundaries
    section_positions = []
    for name, pattern in SECTION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            section_positions.append((match.start(), match.end(), name))

    # Sort by position
    section_positions.sort(key=lambda x: x[0])

    # Extract content between sections
    for i, (start, header_end, name) in enumerate(section_positions):
        # Content starts after the header line
        content_start = text.find("\n", header_end)
        if content_start == -1:
            content_start = header_end

        # Content ends at the next section or end of text
        if i + 1 < len(section_positions):
            content_end = section_positions[i + 1][0]
        else:
            content_end = len(text)

        content = text[content_start:content_end].strip()
        # Remove leading/trailing --- separators
        content = re.sub(r"^---+\s*", "", content)
        content = re.sub(r"\s*---+$", "", content)
        result[name] = content.strip() if content.strip() else None

    # Fill in any sections not found
    for name, _ in SECTION_PATTERNS:
        if name not in result:
            result[name] = None

    # Extract risk level as a clean value
    result["risk_level_value"] = _extract_risk_level(
        result.get("risk_level", "") or ""
    )

    # Extract DPD bucket as a clean value
    result["dpd_bucket_value"] = _extract_dpd_bucket(
        result.get("dpd_bucket", "") or ""
    )

    return result


def _extract_risk_level(text: str) -> Optional[str]:
    """Extract a clean risk level value from the risk section text."""
    text_lower = text.lower()
    if "critical" in text_lower:
        return "Critical"
    elif "high" in text_lower:
        return "High"
    elif "medium" in text_lower or "moderate" in text_lower:
        return "Medium"
    elif "low" in text_lower:
        return "Low"
    return None


def _extract_dpd_bucket(text: str) -> Optional[str]:
    """Extract a clean DPD bucket identifier from the bucket section text."""
    match = re.search(r"bucket\s*(\d+|[xX])", text, re.IGNORECASE)
    if match:
        bucket = match.group(1).upper()
        return f"Bucket {bucket}"
    return None


def format_for_api(result: dict, parsed: dict) -> dict:
    """
    Combine RAG result + parsed sections into a clean API response.

    This is the recommended format for your REST API endpoint.

    Args:
        result: Raw result from get_recommendation()
        parsed: Parsed result from parse_recommendation()

    Returns:
        Clean dict ready for JSON API response
    """
    return {
        # Structured sections (for rendering in UI components)
        "sections": {
            "summary": parsed.get("customer_summary"),
            "dpd_bucket": parsed.get("dpd_bucket"),
            "risk_level": parsed.get("risk_level"),
            "strategy": parsed.get("strategy"),
            "channels": parsed.get("channels"),
            "escalation": parsed.get("escalation"),
            "settlement": parsed.get("settlement"),
            "reasoning": parsed.get("reasoning"),
            "regulatory": parsed.get("regulatory_notes"),
        },
        # Clean values for UI badges/labels
        "labels": {
            "risk": parsed.get("risk_level_value"),
            "bucket": parsed.get("dpd_bucket_value"),
        },
        # Full markdown (for fallback rendering)
        "full_recommendation": result.get("recommendation"),
        # Source references
        "sources": result.get("retrieved_policies", []),
        # Metadata
        "metadata": {
            "cached": result.get("cached", False),
            "pii_removed": result.get("pii_fields_removed", []),
        },
    }
