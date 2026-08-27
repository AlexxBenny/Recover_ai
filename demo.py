"""
Demo script - tests the RAG engine end-to-end.

Usage:
    1. First, generate sample policies:  python scripts/create_sample_policies.py
    2. Create a .env file with your API key (copy from .env.example)
    3. Run this demo:                     python demo.py

This will ingest the policy PDFs, then generate recommendations for
sample defaulter profiles to verify the pipeline works.
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag import ingest_all_policies, get_recommendation, get_store_info


# Sample defaulter profiles matching Lending Club dataset columns
# These are realistic examples based on the actual data distribution
SAMPLE_DEFAULTERS = [
    {
        "name": "Grace Period - Low Risk Borrower",
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
    {
        "name": "Late 31-120 Days - Medium Risk",
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
    {
        "name": "Late 31-120 Days - High Risk with Bad Credit",
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
    {
        "name": "Charged Off - Critical / Write-off",
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
]


def main():
    print("=" * 70)
    print("  RAG Engine Demo - Loan Defaulter Recovery Recommendations")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: Ingest policies
    # ---------------------------------------------------------------
    print("\n[Step 1] Ingesting policy documents...\n")
    try:
        results = ingest_all_policies()
        if not results:
            print("  No PDF files found in policies/ directory.")
            print("  Run 'python scripts/create_sample_policies.py' first.")
            sys.exit(1)

        for r in results:
            if r["success"]:
                print(f"  + {r['source']}: {r['pages']} pages -> {r['chunks_added']} chunks")
            else:
                print(f"  X {r['source']}: FAILED - {r['error']}")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("  Run 'python scripts/create_sample_policies.py' first.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 2: Show store info
    # ---------------------------------------------------------------
    info = get_store_info()
    print(f"\n  Vector Store: {info['total_chunks']} total chunks "
          f"from {len(info['policies_loaded'])} policies")
    print(f"  Embedding model: {info['embedding_model']}")

    # ---------------------------------------------------------------
    # Step 3: Get recommendations
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [Step 2] Generating recommendations for sample defaulters...")
    print("=" * 70)

    for i, defaulter in enumerate(SAMPLE_DEFAULTERS, 1):
        name = defaulter.pop("name")

        # Add a fake PII field to demonstrate sanitization
        defaulter["customer_name"] = "John Doe (FAKE - will be stripped)"
        defaulter["phone_number"] = "555-0123 (FAKE - will be stripped)"

        print(f"\n{'-' * 70}")
        print(f"  Defaulter {i}: {name}")
        print(f"  Loan: Rs {defaulter['loan_amnt']:,.0f} | "
              f"Status: {defaulter['loan_status']} | "
              f"Grade: {defaulter['grade']}{defaulter['sub_grade'][-1]} | "
              f"DTI: {defaulter['dti']}")
        print(f"{'-' * 70}")

        try:
            result = get_recommendation(defaulter)

            # Show PII removal
            if result.get("pii_fields_removed"):
                print(f"\n  [PII] Removed fields: {result['pii_fields_removed']}")

            # Show cache status
            if result.get("cached"):
                print("  [CACHE] Result served from cache")

            print(f"\n  Retrieved {len(result['retrieved_policies'])} policy sections:")
            for p in result["retrieved_policies"]:
                print(f"    - {p['source']} (p.{p['page']}) "
                      f"[relevance: {p['relevance_score']:.3f}]")

            print(f"\n  --- Recommendation ---\n")
            print(result["recommendation"])

        except Exception as e:
            print(f"\n  Error: {e}")

        # Remove fake PII fields and restore name
        defaulter.pop("customer_name", None)
        defaulter.pop("phone_number", None)
        defaulter["name"] = name

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
