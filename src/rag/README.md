# RAG Module — Loan Defaulter Recovery Recommendations

> Policy-grounded, LLM-powered collection strategy recommendations.

---

## Quick Start

```python
from src.rag import ingest_policy, get_recommendation

# 1. Ingest a policy PDF (auto-chunks, embeds, stores in ChromaDB)
ingest_policy("policies/collection_strategy_policy.pdf")

# 2. Get a recommendation for a defaulter (using Lending Club column names)
result = get_recommendation({
    "loan_amnt": 20000,
    "loan_status": "Late (31-120 days)",
    "grade": "B",
    "sub_grade": "B4",
    "annual_inc": 50000,
    "dti": 22.0,
    "home_ownership": "RENT",
    "purpose": "vacation",
    "term": "36 months",
    "int_rate": 11.80,
    "installment": 662.38,
    # ... other Lending Club columns
})

print(result["recommendation"])       # Full LLM recommendation
print(result["retrieved_policies"])    # Source policy sections used
print(result["pii_fields_removed"])    # Any PII that was stripped
print(result["cached"])               # Whether served from cache
```

### Async (for FastAPI / async backends)

```python
from src.rag import aget_recommendation

# Same interface, non-blocking LLM call
result = await aget_recommendation(customer_dict)
```

---

## API Reference

### Ingestion (admin uploads)

| Function | Description |
|---|---|
| `ingest_policy(pdf_path)` | Ingest a single policy PDF. Returns `{success, source, pages, chunks_added, error}` |
| `ingest_all_policies(dir)` | Ingest all PDFs from a directory (default: `./policies/`) |
| `replace_policy(pdf_path)` | Replace an existing policy — deletes old chunks, ingests new PDF, clears cache |

### Recommendation (agent queries)

| Function | Description |
|---|---|
| `get_recommendation(customer)` | Sync — returns recommendation dict |
| `aget_recommendation(customer)` | Async — same interface, for async backends |

### Utilities

| Function | Description |
|---|---|
| `get_store_info()` | Returns vector store stats (chunks, policies, cache info) |
| `clear_cache()` | Clear the response cache |
| `sanitize_customer_data(dict)` | Manually sanitize a customer dict (auto-called internally) |

### Error Types

| Exception | When |
|---|---|
| `PDFLoadError` | PDF is corrupted, password-protected, or has no extractable text |

---

## Architecture

```
Customer Dict ──> Sanitize PII ──> Check Cache ──> Build Query
                                       |               |
                                   [HIT: return]   Retrieval Query
                                                       |
                                                  ChromaDB Search
                                                  (cosine similarity)
                                                       |
                                                 Top-K Policy Chunks
                                                       |
                                              Format Prompt (customer
                                              details + policy context)
                                                       |
                                                   LLM Call
                                                  (Gemini API)
                                                       |
                                              Cache + Return Result
```

### Module Files

| File | Purpose |
|---|---|
| `config.py` | All settings — LLM, embeddings, chunking, cache, limits |
| `pdf_loader.py` | PDF text extraction with error handling (corrupted, password-protected, oversized) |
| `vectorstore.py` | ChromaDB wrapper with `all-MiniLM-L6-v2` embeddings |
| `prompts.py` | Prompt templates + DPD mapping (loan_status → DPD bucket) |
| `rag_engine.py` | Core engine — sync/async, caching, sanitization |
| `sanitize.py` | PII whitelist/blacklist — strips names, phones, IDs before LLM call |
| `cache.py` | In-memory TTL response cache with LRU eviction |
| `mapper.py` | Column name mapper — auto-detects 50+ common DB aliases |
| `response_parser.py` | Parses raw LLM markdown into structured sections for frontend |

---

## Customer Data Format

This module uses **Lending Club dataset column names**. The key fields are:

| Field | Type | Description |
|---|---|---|
| `loan_amnt` | int | Loan amount (INR) |
| `loan_status` | str | `In Grace Period`, `Late (31-120 days)`, `Default`, `Charged Off` |
| `grade` | str | Risk grade A–G |
| `sub_grade` | str | E.g., `A4`, `C5` |
| `annual_inc` | float | Annual income |
| `dti` | float | Debt-to-income ratio |
| `home_ownership` | str | `RENT`, `OWN`, `MORTGAGE` |
| `term` | str | `36 months` or `60 months` |
| `int_rate` | float | Interest rate % |
| `delinq_2yrs` | float | Delinquencies in last 2 years |
| `acc_now_delinq` | float | Currently delinquent accounts |
| `pub_rec` | float | Public records |
| `revol_util` | float | Revolving credit utilization % |

The module also accepts all 45 columns from the dataset — see `prompts.py` for the full list.

### DPD Mapping

Since the dataset doesn't have an explicit DPD column, we derive it from `loan_status`:

| `loan_status` | Estimated DPD | Policy Bucket |
|---|---|---|
| `Current` | 0 | Bucket X |
| `In Grace Period` | ~15 | Bucket 1 (DPD 1–30) |
| `Late (16-30 days)` | ~30 | Bucket 1 (DPD 1–30) |
| `Late (31-120 days)` | ~75 | Bucket 2/3 (DPD 31–90) |
| `Default` | ~150 | Bucket 4 (DPD 91–180, NPA) |
| `Charged Off` | ~180 | Bucket 5 (DPD 180+, Write-off) |

---

## Privacy & PII Handling

### What is automatically stripped before LLM calls

The `sanitize.py` module removes these fields if present in the customer dict:

- **Identity**: `customer_name`, `first_name`, `last_name`, etc.
- **Contact**: `phone`, `email`, `mobile`, etc.
- **Government IDs**: `ssn`, `aadhaar`, `pan`, etc.
- **Full addresses**: `address`, `street`, `city`, `zip`, etc.
- **Account IDs**: `account_number`, `loan_id`, `customer_id`, etc.
- **DOB/Age**: `dob`, `date_of_birth`, `age`

### What IS sent to the LLM

Only anonymized financial metrics (loan amounts, DTI, grades, etc.) and state-level
geography (`addr_state`). The Lending Club dataset itself contains no PII.

### Verification

Every response includes `pii_fields_removed` — a list of field names that were
stripped. Check this in your API response for audit logging.

---

## Caching

Enabled by default. Avoids redundant LLM calls for the same customer profile.

| Setting | Default | Env Variable |
|---|---|---|
| Enabled | `true` | `CACHE_ENABLED` |
| Max entries | 100 | `CACHE_MAX_SIZE` |
| TTL | 3600s (1 hour) | `CACHE_TTL_SECONDS` |

Cache is automatically invalidated when policies are updated via `replace_policy()`.

---

## PDF Upload Error Handling

The module handles these upload failure cases gracefully:

| Scenario | Behavior |
|---|---|
| File not found | Returns `{success: false, error: "PDF not found: ..."}` |
| Not a PDF | Returns error with message about supported formats |
| File too large | Returns error (default limit: 50 MB, configurable via `MAX_PDF_SIZE_MB`) |
| Empty file (0 bytes) | Returns descriptive error |
| Corrupted/unreadable | Returns `PDFLoadError` with details |
| Password-protected | Returns error asking for unprotected version |
| No extractable text (scanned images) | Returns error suggesting OCR is not supported |
| Individual page fails | Skips the page, continues with others, logs warning |

---

## Retrieval Strategy

### Current: Semantic Similarity Search

We use **cosine similarity** with `all-MiniLM-L6-v2` embeddings on ChromaDB.
The retrieval query is built by `build_retrieval_query()` which converts customer
numeric data into semantic keywords optimized for policy matching:

```
Customer: loan_status="Late (31-120 days)", loan_amnt=9100, grade="E"
    ↓ (query augmentation)
Query: "Collection strategy for loan status Late (31-120 days)
        approximately 75 days past due loan amount 9100 risk grade E
        hard collection demand notice intensive follow-up"
```

This query augmentation bridges the gap between numeric customer data and
semantic policy text, achieving **100% retrieval precision** in evaluation.

### Why Not Hybrid Search?

Hybrid search (combining semantic + keyword/BM25) is useful when:
- The corpus is very large (1000s of documents)
- Exact numeric matching is critical at the retrieval level
- Semantic similarity alone can't distinguish between similar sections

**For our use case, pure semantic search works** because:
1. We search **policy documents**, not customer data — the corpus is small and focused
2. The **query augmentation** layer converts numbers into semantic keywords
3. **Exact numeric matching happens at the LLM level** — the LLM sees both the
   customer's numbers and the policy text, and matches the tiers correctly
4. Our evaluation shows **100% retrieval accuracy** across all 5 test cases

### Future Enhancement: Hybrid Search

If the policy corpus grows significantly, consider adding BM25 keyword search
alongside semantic search. ChromaDB supports `where` filters that could be used
to pre-filter by DPD range metadata before semantic ranking.

---

## Evaluation

### Run the evaluation

```bash
# Retrieval only (no API key needed, instant)
python scripts/evaluate_rag.py --mode retrieval

# Full evaluation (needs GOOGLE_API_KEY in .env)
python scripts/evaluate_rag.py --mode full

# Both
python scripts/evaluate_rag.py --mode both
```

### Latest Results (5 test cases, 7 dimensions)

| Dimension | Score | Description |
|---|---|---|
| Retrieval | 100% | Correct policy sources retrieved |
| DPD Bucket | 100% | Correct DPD bucket identified |
| Risk Level | 100% | Correct priority classification |
| Strategy | 91% | Strategy aligns with policy rules |
| Settlement | 100% | Correct OTS discount percentages |
| Escalation | 60% | Correct team but uses descriptive names vs exact "Level N" |
| Channels | 100% | Correct communication channels |
| **Overall** | **94.3%** | Weighted average |

---

## Configuration

All settings are configurable via environment variables (`.env` file):

```env
# LLM
GOOGLE_API_KEY=your_key_here
LLM_PROVIDER=google              # google | openai | ollama
LLM_MODEL=gemini-3.6-flash
LLM_TEMPERATURE=0.2

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5

# Cache
CACHE_ENABLED=true
CACHE_MAX_SIZE=100
CACHE_TTL_SECONDS=3600

# Upload Limits
MAX_PDF_SIZE_MB=50
```

---

## Integration Guide (for backend teammate)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up .env

```bash
cp .env.example .env
# Add your GOOGLE_API_KEY
```

### 3. Column Mapping (if your DB uses different names)

Your database probably doesn't use Lending Club column names. Use `ColumnMapper`:

```python
from src.rag import ColumnMapper

# Option A: Auto-detect common names (works out of the box)
mapper = ColumnMapper()

# Option B: Add custom mappings for your specific DB schema
mapper = ColumnMapper(custom_mapping={
    "customer_loan_amount": "loan_amnt",
    "customer_income": "annual_inc",
    "account_status": "loan_status",
})

# Map any DB row — unknown columns pass through, PII gets stripped later
db_row = fetch_from_database(customer_id)
mapped = mapper.map(db_row)

# Validate before calling (checks critical fields are present)
validation = mapper.validate(db_row)
if not validation["ready"]:
    print(f"Missing: {validation['missing_critical']}")
```

Auto-detected aliases include: `loan_amount`, `annual_income`, `interest_rate`,
`debt_to_income`, `status`, `employment_length`, `salary`, and 50+ more.
See `mapper.py` for the full list.

### 4. Full Integration Flow

```
Agent clicks customer profile in dashboard
    |
    v
Backend queries DB for customer row
    |
    v
ColumnMapper.map(db_row)          <-- maps to LC column names
    |
    v
get_recommendation(mapped_dict)   <-- PII auto-stripped, cache checked
    |
    v
parse_recommendation(result)      <-- raw markdown → structured sections
    |
    v
format_for_api(result, parsed)    <-- clean JSON for frontend
    |
    v
Frontend renders:
  - Risk badge (Critical/High/Medium/Low)
  - Strategy card
  - Settlement terms panel
  - Source references
```

### 5. FastAPI Example (complete)

```python
from fastapi import FastAPI, UploadFile, HTTPException
from src.rag import (
    ColumnMapper,
    aget_recommendation,
    ingest_policy,
    replace_policy,
    get_store_info,
    parse_recommendation,
    format_for_api,
)

app = FastAPI()

# Initialize column mapper once at startup
mapper = ColumnMapper(custom_mapping={
    # Add your DB-specific column names here
})


# --- Policy Management (Admin) ---

@app.post("/api/policies/upload")
async def upload_policy(file: UploadFile):
    """Admin uploads a new policy PDF."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    path = f"policies/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    result = ingest_policy(path)
    if not result["success"]:
        raise HTTPException(422, result["error"])
    return result


@app.put("/api/policies/{filename}")
async def update_policy(filename: str, file: UploadFile):
    """Admin replaces an existing policy."""
    path = f"policies/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    return replace_policy(path)


# --- Recommendations (Collection Agent) ---

@app.post("/api/recommend")
async def recommend(customer: dict):
    """Get recommendation for a customer profile."""
    # Step 1: Map DB columns to LC names
    mapped = mapper.map(customer)

    # Step 2: Get recommendation (PII stripped, cache checked)
    result = await aget_recommendation(mapped)

    # Step 3: Parse into structured sections
    parsed = parse_recommendation(result["recommendation"])

    # Step 4: Return clean API response
    return format_for_api(result, parsed)


@app.get("/api/store/info")
def store_info():
    """Dashboard: show loaded policies and cache stats."""
    return get_store_info()
```

### 6. What the API Returns

The `format_for_api()` function returns this structure:

```json
{
    "sections": {
        "summary": "Borrower has a Rs 20,000 loan, Grade B4...",
        "dpd_bucket": "Bucket 3: DPD 61-90 (Serious Delinquency)",
        "risk_level": "High",
        "strategy": "1. Intensive tele-calling...",
        "channels": "Primary: Phone Call, Secondary: Email",
        "escalation": "Yes - escalate to Hard Collection Team",
        "settlement": "OTS eligible with up to 10% discount",
        "reasoning": "Per collection_strategy_policy.pdf, Page 2...",
        "regulatory": "Contact hours: 8 AM - 7 PM only..."
    },
    "labels": {
        "risk": "High",
        "bucket": "Bucket 3"
    },
    "full_recommendation": "... full markdown text ...",
    "sources": [
        {"source": "collection_strategy_policy.pdf", "page": 2, "relevance_score": 0.67}
    ],
    "metadata": {
        "cached": false,
        "pii_removed": ["customer_name", "phone_number"]
    }
}
```

Use `sections` for individual UI components, `labels` for badges/chips,
and `full_recommendation` as a markdown fallback.

### 7. What You DON'T Need to Handle

| Concern | Handled by |
|---|---|
| Renaming DB columns | `ColumnMapper` |
| Stripping customer PII | `sanitize.py` (auto) |
| Caching repeated queries | `cache.py` (auto) |
| DPD calculation from loan_status | `prompts.py` (auto) |
| PDF parsing and chunking | `pdf_loader.py` + `vectorstore.py` |
| LLM prompt engineering | `prompts.py` |
| Policy versioning | `replace_policy()` (auto-invalidates cache) |
| Corrupted PDF uploads | Returns `{success: false, error: "..."}` |

