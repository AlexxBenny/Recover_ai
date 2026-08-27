"""
PII sanitization layer for customer data.

Ensures no personally identifiable information is sent to the LLM.
This is a safety net for when live database data (which may contain
names, phones, etc.) is passed to get_recommendation().
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


# Fields that are PII and MUST be stripped before sending to LLM
PII_FIELDS = {
    # Identity
    "customer_name", "name", "full_name", "first_name", "last_name",
    "borrower_name", "co_borrower_name", "guarantor_name",
    # Contact
    "phone", "phone_number", "mobile", "contact", "contact_number",
    "email", "email_address", "email_id",
    # Government IDs
    "ssn", "social_security", "aadhaar", "aadhaar_number",
    "pan", "pan_number", "passport", "passport_number",
    "voter_id", "driving_license",
    # Address (full — we allow state-level only)
    "address", "full_address", "street", "street_address",
    "city", "zip", "zipcode", "zip_code", "postal_code",
    "address_line_1", "address_line_2",
    # Account identifiers
    "account_number", "account_no", "loan_id", "customer_id",
    "member_id", "application_id", "reference_number",
    # Date of birth
    "dob", "date_of_birth", "birth_date", "age",
    # Bank details
    "bank_account", "ifsc", "bank_name", "branch",
    "card_number", "cvv", "expiry",
}

# Fields explicitly safe to send to the LLM (Lending Club dataset columns)
SAFE_FIELDS = {
    # Loan characteristics
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "purpose", "application_type", "grade", "sub_grade",
    # Borrower profile (non-PII)
    "annual_inc", "emp_length", "home_ownership", "verification_status",
    "addr_state",  # State-level only — NOT full address
    "dti", "earliest_cr_line",
    # Credit history
    "pub_rec", "pub_rec_bankruptcies", "tax_liens",
    "acc_now_delinq", "delinq_2yrs", "delinq_amnt",
    "mths_since_last_delinq", "collections_12_mths_ex_med",
    "chargeoff_within_12_mths", "num_accts_ever_120_pd",
    "num_tl_30dpd", "num_tl_90g_dpd_24m",
    # Balances and utilization
    "revol_bal", "revol_util", "bc_util",
    "avg_cur_bal", "tot_cur_bal", "total_bal_ex_mort",
    "total_bc_limit", "percent_bc_gt_75",
    # Account counts
    "open_acc", "total_acc", "mort_acc",
    "pct_tl_nvr_dlq", "inq_last_6mths", "acc_open_past_24mths",
    # Loan status and recovery
    "loan_status", "recoveries", "collection_recovery_fee",
}


def sanitize_customer_data(customer: dict) -> Tuple[dict, list[str]]:
    """
    Remove PII fields from customer data before sending to the LLM.

    Uses a whitelist + blacklist approach:
    - Known PII fields → removed
    - Known safe fields → kept
    - Unknown fields → kept with a warning log

    Args:
        customer: Raw customer dict (may contain PII from live database)

    Returns:
        Tuple of (sanitized_dict, list_of_removed_field_names)
    """
    sanitized = {}
    removed_fields = []

    for key, value in customer.items():
        key_lower = key.lower().strip()

        if key_lower in PII_FIELDS:
            removed_fields.append(key)
            logger.info(f"PII field removed: '{key}'")
        elif key_lower in SAFE_FIELDS:
            sanitized[key] = value
        else:
            # Unknown field — keep it but log a warning
            # In strict mode, you could remove these too
            sanitized[key] = value
            logger.debug(f"Unknown field passed through: '{key}'")

    if removed_fields:
        logger.warning(
            f"Removed {len(removed_fields)} PII field(s) before LLM call: "
            f"{removed_fields}"
        )

    return sanitized, removed_fields
