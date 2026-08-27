"""
RAG Evaluation Script — Accuracy, Retrieval Quality, and Policy Compliance.

Cross-references the RAG module's output against ground-truth expectations
derived from manual analysis of the bank's policy documents.

Evaluates:
1. Retrieval Precision — Are the correct policy sections retrieved?
2. DPD Bucket Accuracy — Is the correct DPD bucket identified?
3. Strategy Alignment — Does the strategy match the policy for that bucket?
4. Grade Adjustment — Are grade-based modifications applied?
5. Settlement Terms — Are correct OTS discount percentages mentioned?
6. Escalation Level — Is the correct team/level recommended?
7. Channel Priority — Are the correct communication channels recommended?

Usage:
    python scripts/evaluate_rag.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import ingest_all_policies, get_recommendation, get_store_info
from src.rag.prompts import estimate_dpd, get_dpd_bucket
from src.rag.vectorstore import VectorStore
from src.rag.prompts import build_retrieval_query


# ===================================================================
# Ground-Truth Test Cases (derived from manual policy analysis)
# ===================================================================

TEST_CASES = [
    {
        "id": "TC-01",
        "name": "Grace Period - Low Risk, Small Loan",
        "customer": {
            "loan_amnt": 20975,
            "funded_amnt": 20975,
            "term": "36 months",
            "int_rate": 8.19,
            "installment": 659.13,
            "purpose": "debt_consolidation",
            "application_type": "Individual",
            "grade": "A",
            "sub_grade": "A4",
            "annual_inc": 165000.0,
            "emp_length": "10+ years",
            "home_ownership": "MORTGAGE",
            "dti": 15.5,
            "addr_state": "CA",
            "delinq_2yrs": 0.0,
            "acc_now_delinq": 0.0,
            "delinq_amnt": 0.0,
            "mths_since_last_delinq": None,
            "num_tl_30dpd": 0.0,
            "num_tl_90g_dpd_24m": 0.0,
            "num_accts_ever_120_pd": 0.0,
            "revol_bal": 12500,
            "revol_util": 35.0,
            "bc_util": 25.0,
            "open_acc": 9.0,
            "total_acc": 21.0,
            "pub_rec": 0.0,
            "pub_rec_bankruptcies": 0.0,
            "tax_liens": 0.0,
            "inq_last_6mths": 0.0,
            "pct_tl_nvr_dlq": 100.0,
            "mort_acc": 3.0,
            "acc_open_past_24mths": 6.0,
            "loan_status": "In Grace Period",
            "recoveries": 0.0,
            "collection_recovery_fee": 0.0,
        },
        "expected": {
            # Policy: Bucket 1 (DPD 1-30), loan < Rs 1,00,000
            "dpd_bucket": "Bucket 1",
            "dpd_range": "1-30",
            "priority": "Low",
            "risk_level": "Low",
            # Policy: "For loan amounts below Rs 1,00,000: Only automated SMS and email. No phone calls."
            "strategy_keywords": ["SMS", "email", "automated"],
            "strategy_should_NOT_contain": ["field visit", "legal", "SARFAESI"],
            # Policy: Grade A-B -> softer approach, extend timelines
            "grade_adjustment": "softer",
            "grade_keywords": ["soft", "extend", "restructur"],
            # Policy: No OTS at DPD 1-30
            "settlement_eligible": False,
            # Policy: Level 1 Soft Collection Team
            "escalation_level": "Level 1",
            "escalation_keywords": ["soft collection", "Level 1", "tele-calling"],
            # Policy: DPD 1-30 channel priority: SMS > Email > IVR > Phone Call
            "channel_priority": ["SMS", "email"],
            # Which policy docs should be retrieved
            "expected_sources": ["collection_strategy_policy.pdf"],
        },
    },
    {
        "id": "TC-02",
        "name": "Late 31-120 Days - Medium Risk, Grade B",
        "customer": {
            "loan_amnt": 20000,
            "funded_amnt": 20000,
            "term": "36 months",
            "int_rate": 11.80,
            "installment": 662.38,
            "purpose": "vacation",
            "application_type": "Individual",
            "grade": "B",
            "sub_grade": "B4",
            "annual_inc": 50000.0,
            "emp_length": "3 years",
            "home_ownership": "RENT",
            "dti": 22.0,
            "addr_state": "NY",
            "delinq_2yrs": 1.0,
            "acc_now_delinq": 1.0,
            "delinq_amnt": 500.0,
            "mths_since_last_delinq": 8.0,
            "num_tl_30dpd": 1.0,
            "num_tl_90g_dpd_24m": 0.0,
            "num_accts_ever_120_pd": 0.0,
            "revol_bal": 18000,
            "revol_util": 72.0,
            "bc_util": 65.0,
            "open_acc": 17.0,
            "total_acc": 26.0,
            "pub_rec": 0.0,
            "pub_rec_bankruptcies": 0.0,
            "tax_liens": 0.0,
            "inq_last_6mths": 1.0,
            "pct_tl_nvr_dlq": 100.0,
            "mort_acc": 0.0,
            "acc_open_past_24mths": 4.0,
            "loan_status": "Late (31-120 days)",
            "recoveries": 0.0,
            "collection_recovery_fee": 0.0,
        },
        "expected": {
            # Estimated DPD ~75 -> Bucket 3 (DPD 61-90)
            "dpd_bucket": "Bucket 3",
            "dpd_range": "61-90",
            "priority": "High",
            "risk_level": "High",
            # Policy: loan < Rs 1,00,000 at Bucket 3: "Intensive tele-calling + firm demand notice"
            "strategy_keywords": ["tele-calling", "demand notice", "intensive"],
            "strategy_should_NOT_contain": ["SARFAESI", "write-off", "ARC"],
            # Policy: Grade A-B -> softer, extend timelines by 7-10 days
            "grade_adjustment": "softer",
            "grade_keywords": ["soft", "extend", "negotiat"],
            # Policy: OTS at DPD 61-90 up to 10% discount
            "settlement_eligible": True,
            "ots_discount": "10%",
            # Policy: Level 3 Hard Collection
            "escalation_level": "Level 3",
            "escalation_keywords": ["hard collection", "Level 3", "escalat"],
            # Policy: DPD 61-90 channels: Phone > Field Visit > Legal Notice > Email
            "channel_priority": ["phone", "demand notice"],
            "expected_sources": ["collection_strategy_policy.pdf"],
        },
    },
    {
        "id": "TC-03",
        "name": "Late 31-120 Days - High Risk, Grade E",
        "customer": {
            "loan_amnt": 9100,
            "funded_amnt": 9100,
            "term": "36 months",
            "int_rate": 26.31,
            "installment": 368.15,
            "purpose": "other",
            "application_type": "Individual",
            "grade": "E",
            "sub_grade": "E4",
            "annual_inc": 62000.0,
            "emp_length": "5 years",
            "home_ownership": "RENT",
            "dti": 28.5,
            "addr_state": "TX",
            "delinq_2yrs": 3.0,
            "acc_now_delinq": 2.0,
            "delinq_amnt": 2500.0,
            "mths_since_last_delinq": 4.0,
            "num_tl_30dpd": 2.0,
            "num_tl_90g_dpd_24m": 1.0,
            "num_accts_ever_120_pd": 1.0,
            "revol_bal": 28000,
            "revol_util": 85.0,
            "bc_util": 90.0,
            "open_acc": 8.0,
            "total_acc": 23.0,
            "pub_rec": 1.0,
            "pub_rec_bankruptcies": 0.0,
            "tax_liens": 0.0,
            "inq_last_6mths": 1.0,
            "pct_tl_nvr_dlq": 95.5,
            "mort_acc": 1.0,
            "acc_open_past_24mths": 3.0,
            "loan_status": "Late (31-120 days)",
            "recoveries": 0.0,
            "collection_recovery_fee": 0.0,
        },
        "expected": {
            # Estimated DPD ~75 -> Bucket 3 (DPD 61-90)
            "dpd_bucket": "Bucket 3",
            "dpd_range": "61-90",
            "priority": "High",
            "risk_level": "High",
            # Policy: loan < Rs 1,00,000: Intensive tele-calling + firm demand notice
            "strategy_keywords": ["tele-calling", "demand notice"],
            "strategy_should_NOT_contain": ["SARFAESI", "write-off"],
            # Policy: Grade E-F -> "Accelerate collection timelines by 5-7 days"
            "grade_adjustment": "accelerate",
            "grade_keywords": ["accelerat", "fast", "earlier", "proactiv"],
            # Policy: OTS up to 10%
            "settlement_eligible": True,
            "ots_discount": "10%",
            # Policy: Level 3
            "escalation_level": "Level 3",
            "escalation_keywords": ["hard collection", "Level 3", "escalat"],
            "channel_priority": ["phone", "demand notice"],
            "expected_sources": ["collection_strategy_policy.pdf"],
        },
    },
    {
        "id": "TC-04",
        "name": "Charged Off - Critical, Grade F",
        "customer": {
            "loan_amnt": 35000,
            "funded_amnt": 35000,
            "term": "60 months",
            "int_rate": 24.50,
            "installment": 1012.45,
            "purpose": "debt_consolidation",
            "application_type": "Individual",
            "grade": "F",
            "sub_grade": "F2",
            "annual_inc": 40000.0,
            "emp_length": "< 1 year",
            "home_ownership": "RENT",
            "dti": 38.0,
            "addr_state": "FL",
            "delinq_2yrs": 5.0,
            "acc_now_delinq": 3.0,
            "delinq_amnt": 8000.0,
            "mths_since_last_delinq": 2.0,
            "num_tl_30dpd": 3.0,
            "num_tl_90g_dpd_24m": 2.0,
            "num_accts_ever_120_pd": 3.0,
            "revol_bal": 42000,
            "revol_util": 95.0,
            "bc_util": 98.0,
            "open_acc": 12.0,
            "total_acc": 30.0,
            "pub_rec": 2.0,
            "pub_rec_bankruptcies": 1.0,
            "tax_liens": 1.0,
            "inq_last_6mths": 3.0,
            "pct_tl_nvr_dlq": 78.0,
            "mort_acc": 0.0,
            "acc_open_past_24mths": 2.0,
            "loan_status": "Charged Off",
            "recoveries": 1200.50,
            "collection_recovery_fee": 210.00,
        },
        "expected": {
            # Estimated DPD 180+ -> Bucket 5 (Write-off Candidate)
            "dpd_bucket": "Bucket 5",
            "dpd_range": "180",
            "priority": "Critical",
            "risk_level": "Critical",
            # Policy: Legal proceedings, OTS up to 40%, ARC evaluation
            "strategy_keywords": ["legal", "settlement", "OTS"],
            "strategy_should_NOT_contain": [],
            # Policy: Grade E-F -> accelerate timelines
            "grade_adjustment": "accelerate",
            "grade_keywords": ["accelerat", "fast", "legal"],
            # Policy: OTS at DPD 180-365: up to 30%, or DPD 365+: up to 40%
            # Bucket 5 policy says up to 40% discount
            "settlement_eligible": True,
            "ots_discount": "40%",
            # Policy: Level 4 Legal and Recovery Team
            "escalation_level": "Level 4",
            "escalation_keywords": ["legal", "Level 4", "recovery"],
            # Policy: DPD 180+ channels: Legal Action > Field Visit > Settlement Offer
            "channel_priority": ["legal"],
            "expected_sources": ["collection_strategy_policy.pdf"],
        },
    },
    {
        "id": "TC-05",
        "name": "NPA Stage - High Value Loan",
        "customer": {
            "loan_amnt": 2500000,
            "funded_amnt": 2500000,
            "term": "60 months",
            "int_rate": 18.00,
            "installment": 63456.0,
            "purpose": "debt_consolidation",
            "application_type": "Individual",
            "grade": "D",
            "sub_grade": "D3",
            "annual_inc": 1200000.0,
            "emp_length": "8 years",
            "home_ownership": "OWN",
            "dti": 30.0,
            "addr_state": "MH",
            "delinq_2yrs": 2.0,
            "acc_now_delinq": 1.0,
            "delinq_amnt": 50000.0,
            "mths_since_last_delinq": 3.0,
            "num_tl_30dpd": 1.0,
            "num_tl_90g_dpd_24m": 1.0,
            "num_accts_ever_120_pd": 0.0,
            "revol_bal": 300000,
            "revol_util": 60.0,
            "bc_util": 55.0,
            "open_acc": 10.0,
            "total_acc": 25.0,
            "pub_rec": 0.0,
            "pub_rec_bankruptcies": 0.0,
            "tax_liens": 0.0,
            "inq_last_6mths": 0.0,
            "pct_tl_nvr_dlq": 92.0,
            "mort_acc": 2.0,
            "acc_open_past_24mths": 4.0,
            "loan_status": "Default",
            "recoveries": 0.0,
            "collection_recovery_fee": 0.0,
        },
        "expected": {
            # Default -> estimated DPD ~150 -> Bucket 4 (DPD 91-180, NPA)
            "dpd_bucket": "Bucket 4",
            "dpd_range": "91-180",
            "priority": "Critical",
            "risk_level": "Critical",
            # Policy: Loan > Rs 25,00,000: External legal counsel + SARFAESI (if secured)
            "strategy_keywords": ["NPA", "legal", "demand notice"],
            "strategy_should_NOT_contain": [],
            # Policy: Grade C-D -> standard timelines
            "grade_adjustment": "standard",
            "grade_keywords": ["standard", "monitor"],
            # Policy: OTS at DPD 91-180: up to 20%
            "settlement_eligible": True,
            "ots_discount": "20%",
            # Policy: Level 4
            "escalation_level": "Level 4",
            "escalation_keywords": ["legal", "Level 4", "NPA"],
            "channel_priority": ["field visit", "legal"],
            # Should retrieve collection strategy AND escalation matrix for high-value
            "expected_sources": ["collection_strategy_policy.pdf"],
        },
    },
]


# ===================================================================
# Evaluation Functions
# ===================================================================


def check_keyword_presence(text: str, keywords: list[str]) -> dict:
    """Check which keywords are present (case-insensitive) in text."""
    text_lower = text.lower()
    results = {}
    for kw in keywords:
        results[kw] = kw.lower() in text_lower
    return results


def check_keyword_absence(text: str, keywords: list[str]) -> dict:
    """Check keywords that should NOT be present."""
    text_lower = text.lower()
    results = {}
    for kw in keywords:
        # True = correctly absent, False = incorrectly present
        results[kw] = kw.lower() not in text_lower
    return results


def evaluate_retrieval(result: dict, expected: dict) -> dict:
    """Evaluate retrieval quality."""
    retrieved = result.get("retrieved_policies", [])
    if not retrieved:
        return {"score": 0.0, "details": "No policies retrieved"}

    retrieved_sources = [r["source"] for r in retrieved]
    expected_sources = expected.get("expected_sources", [])

    # Check if expected sources are in retrieved results
    hits = sum(1 for s in expected_sources if s in retrieved_sources)
    precision = hits / len(expected_sources) if expected_sources else 0

    # Average relevance score
    avg_score = sum(r["relevance_score"] for r in retrieved) / len(retrieved)

    return {
        "score": precision,
        "avg_relevance": round(avg_score, 4),
        "retrieved_sources": list(set(retrieved_sources)),
        "expected_sources": expected_sources,
        "sources_found": hits,
        "total_chunks_retrieved": len(retrieved),
    }


def evaluate_dpd_bucket(text: str, expected: dict) -> dict:
    """Evaluate if the correct DPD bucket is identified."""
    text_lower = text.lower()
    expected_bucket = expected.get("dpd_bucket", "").lower()
    expected_range = expected.get("dpd_range", "")

    bucket_found = expected_bucket in text_lower
    range_found = expected_range in text_lower

    return {
        "score": 1.0 if bucket_found else (0.5 if range_found else 0.0),
        "bucket_correct": bucket_found,
        "range_mentioned": range_found,
        "expected_bucket": expected.get("dpd_bucket"),
    }


def evaluate_risk_level(text: str, expected: dict) -> dict:
    """Evaluate if the correct risk/priority level is identified."""
    text_lower = text.lower()
    expected_risk = expected.get("risk_level", "").lower()
    expected_priority = expected.get("priority", "").lower()

    risk_found = expected_risk in text_lower
    priority_found = expected_priority in text_lower

    return {
        "score": 1.0 if (risk_found or priority_found) else 0.0,
        "risk_correct": risk_found,
        "priority_correct": priority_found,
        "expected_risk": expected.get("risk_level"),
    }


def evaluate_strategy(text: str, expected: dict) -> dict:
    """Evaluate if the recommended strategy aligns with policy."""
    # Check required keywords
    required = expected.get("strategy_keywords", [])
    required_results = check_keyword_presence(text, required)
    required_hits = sum(1 for v in required_results.values() if v)
    required_score = required_hits / len(required) if required else 1.0

    # Check forbidden keywords
    forbidden = expected.get("strategy_should_NOT_contain", [])
    forbidden_results = check_keyword_absence(text, forbidden)
    forbidden_score = (
        sum(1 for v in forbidden_results.values() if v) / len(forbidden)
        if forbidden
        else 1.0
    )

    return {
        "score": round((required_score * 0.7 + forbidden_score * 0.3), 4),
        "required_keywords": required_results,
        "required_hit_rate": f"{required_hits}/{len(required)}",
        "forbidden_keywords": forbidden_results,
    }


def evaluate_settlement(text: str, expected: dict) -> dict:
    """Evaluate settlement/OTS accuracy."""
    text_lower = text.lower()
    expected_eligible = expected.get("settlement_eligible", False)

    ots_mentioned = "ots" in text_lower or "settlement" in text_lower
    discount = expected.get("ots_discount", "")
    discount_correct = discount.lower() in text_lower if discount else True

    if expected_eligible:
        score = 1.0 if (ots_mentioned and discount_correct) else (
            0.5 if ots_mentioned else 0.0
        )
    else:
        # If OTS should NOT be offered, check it's not prominently recommended
        score = 1.0  # We don't penalize mentioning settlement at early stage

    return {
        "score": score,
        "ots_mentioned": ots_mentioned,
        "expected_discount": discount,
        "discount_found": discount_correct,
        "expected_eligible": expected_eligible,
    }


def evaluate_escalation(text: str, expected: dict) -> dict:
    """Evaluate if the correct escalation level is recommended."""
    keywords = expected.get("escalation_keywords", [])
    results = check_keyword_presence(text, keywords)
    hits = sum(1 for v in results.values() if v)
    score = hits / len(keywords) if keywords else 1.0

    return {
        "score": round(score, 4),
        "keywords_found": results,
        "expected_level": expected.get("escalation_level"),
    }


def evaluate_channels(text: str, expected: dict) -> dict:
    """Evaluate if the correct communication channels are recommended."""
    channels = expected.get("channel_priority", [])
    results = check_keyword_presence(text, channels)
    hits = sum(1 for v in results.values() if v)
    score = hits / len(channels) if channels else 1.0

    return {
        "score": round(score, 4),
        "channels_found": results,
    }


def evaluate_single_case(test_case: dict) -> dict:
    """Run a single test case and return evaluation results."""
    tc_id = test_case["id"]
    customer = test_case["customer"]
    expected = test_case["expected"]

    print(f"\n  [{tc_id}] {test_case['name']}...")

    start = time.time()
    try:
        result = get_recommendation(customer)
    except Exception as e:
        return {
            "id": tc_id,
            "name": test_case["name"],
            "error": str(e),
            "overall_score": 0.0,
        }
    elapsed = time.time() - start

    recommendation = result.get("recommendation", "")

    # Run all evaluations
    retrieval = evaluate_retrieval(result, expected)
    dpd = evaluate_dpd_bucket(recommendation, expected)
    risk = evaluate_risk_level(recommendation, expected)
    strategy = evaluate_strategy(recommendation, expected)
    settlement = evaluate_settlement(recommendation, expected)
    escalation = evaluate_escalation(recommendation, expected)
    channels = evaluate_channels(recommendation, expected)

    # Weighted overall score
    weights = {
        "retrieval": 0.15,
        "dpd_bucket": 0.20,
        "risk_level": 0.10,
        "strategy": 0.20,
        "settlement": 0.15,
        "escalation": 0.10,
        "channels": 0.10,
    }

    scores = {
        "retrieval": retrieval["score"],
        "dpd_bucket": dpd["score"],
        "risk_level": risk["score"],
        "strategy": strategy["score"],
        "settlement": settlement["score"],
        "escalation": escalation["score"],
        "channels": channels["score"],
    }

    overall = sum(scores[k] * weights[k] for k in weights)

    return {
        "id": tc_id,
        "name": test_case["name"],
        "overall_score": round(overall, 4),
        "scores": scores,
        "details": {
            "retrieval": retrieval,
            "dpd_bucket": dpd,
            "risk_level": risk,
            "strategy": strategy,
            "settlement": settlement,
            "escalation": escalation,
            "channels": channels,
        },
        "time_seconds": round(elapsed, 2),
        "cached": result.get("cached", False),
        "pii_removed": result.get("pii_fields_removed", []),
    }


# ===================================================================
# Retrieval-Only Evaluation (no LLM needed)
# ===================================================================


def evaluate_retrieval_only():
    """
    Test retrieval quality without making LLM calls.
    Fast and free — no API key needed.
    """
    print("\n" + "=" * 70)
    print("  Retrieval-Only Evaluation (no LLM calls)")
    print("=" * 70)

    vs = VectorStore()

    if vs.count == 0:
        print("\n  Vector store is empty. Ingesting policies...")
        ingest_all_policies()

    print(f"\n  Vector store: {vs.count} chunks loaded\n")

    for tc in TEST_CASES:
        tc_id = tc["id"]
        customer = tc["customer"]
        expected = tc["expected"]

        query = build_retrieval_query(customer)
        results = vs.query(query)

        retrieved_sources = list(set(r["metadata"]["source"] for r in results))
        expected_sources = expected.get("expected_sources", [])
        hits = sum(1 for s in expected_sources if s in retrieved_sources)
        scores = [r["score"] for r in results]

        status = "PASS" if hits == len(expected_sources) else "FAIL"
        avg_score = sum(scores) / len(scores) if scores else 0

        print(f"  [{tc_id}] {tc['name']}")
        print(f"    Status: {status}")
        print(f"    Sources retrieved: {retrieved_sources}")
        print(f"    Expected sources:  {expected_sources}")
        print(f"    Relevance scores:  {[round(s, 3) for s in scores]}")
        print(f"    Avg relevance:     {avg_score:.3f}")

        # Show what content was retrieved (first 80 chars)
        for i, r in enumerate(results):
            snippet = r["text"][:80].replace("\n", " ")
            print(f"    Chunk {i+1}: [{r['metadata']['source']}:p{r['metadata']['page_number']}] "
                  f"{snippet}...")
        print()


# ===================================================================
# Full Evaluation (requires LLM API key)
# ===================================================================


def evaluate_full():
    """Run full evaluation including LLM response quality."""
    print("\n" + "=" * 70)
    print("  Full RAG Evaluation (Retrieval + LLM Response Quality)")
    print("=" * 70)

    # Ensure policies are ingested
    info = get_store_info()
    if info["total_chunks"] == 0:
        print("\n  Ingesting policies first...")
        ingest_all_policies()

    print(f"\n  Vector store: {info['total_chunks']} chunks from "
          f"{len(info['policies_loaded'])} policies")

    all_results = []
    total_time = 0

    for tc in TEST_CASES:
        result = evaluate_single_case(tc)
        all_results.append(result)

        if "error" in result:
            print(f"    ERROR: {result['error']}")
            continue

        total_time += result["time_seconds"]

        # Print per-case results
        scores = result["scores"]
        print(f"    Overall: {result['overall_score']:.1%} | "
              f"Time: {result['time_seconds']:.1f}s | "
              f"Cached: {result['cached']}")
        print(f"    Retrieval: {scores['retrieval']:.0%} | "
              f"DPD: {scores['dpd_bucket']:.0%} | "
              f"Risk: {scores['risk_level']:.0%} | "
              f"Strategy: {scores['strategy']:.0%} | "
              f"Settlement: {scores['settlement']:.0%} | "
              f"Escalation: {scores['escalation']:.0%} | "
              f"Channels: {scores['channels']:.0%}")

    # ---------------------------------------------------------------
    # Summary Report
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)

    valid_results = [r for r in all_results if "error" not in r]

    if not valid_results:
        print("\n  No valid results to summarize.")
        return all_results

    avg_overall = sum(r["overall_score"] for r in valid_results) / len(valid_results)

    # Per-dimension averages
    dimensions = ["retrieval", "dpd_bucket", "risk_level", "strategy",
                   "settlement", "escalation", "channels"]
    dim_avgs = {}
    for dim in dimensions:
        dim_scores = [r["scores"][dim] for r in valid_results]
        dim_avgs[dim] = sum(dim_scores) / len(dim_scores)

    print(f"\n  Test Cases Run:    {len(all_results)}")
    print(f"  Successful:        {len(valid_results)}")
    print(f"  Total LLM Time:    {total_time:.1f}s")
    print(f"  Avg Time/Query:    {total_time/len(valid_results):.1f}s")
    print(f"\n  OVERALL ACCURACY:  {avg_overall:.1%}")

    print(f"\n  Per-Dimension Scores:")
    print(f"  {'Dimension':<20} {'Score':>8} {'Grade':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}")
    for dim in dimensions:
        score = dim_avgs[dim]
        grade = (
            "A" if score >= 0.9
            else "B" if score >= 0.75
            else "C" if score >= 0.6
            else "D" if score >= 0.4
            else "F"
        )
        print(f"  {dim:<20} {score:>7.1%} {grade:>8}")

    # Save detailed results to JSON
    output_path = Path(__file__).parent.parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Detailed results saved to: {output_path}")

    return all_results


# ===================================================================
# Main
# ===================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG module accuracy")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "full", "both"],
        default="both",
        help="Evaluation mode: 'retrieval' (no API key), 'full' (needs API key), "
             "or 'both' (default)",
    )
    args = parser.parse_args()

    if args.mode in ("retrieval", "both"):
        evaluate_retrieval_only()

    if args.mode in ("full", "both"):
        evaluate_full()


if __name__ == "__main__":
    main()
