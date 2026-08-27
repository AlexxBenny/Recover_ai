"""
Column mapper for integrating with external databases.

Maps external database column names to the Lending Club column names
expected by the RAG module. This eliminates the need for the backend
teammate to manually rename fields before calling get_recommendation().

Usage:
    from src.rag.mapper import ColumnMapper

    # Define your DB's column names once
    mapper = ColumnMapper({
        "loan_amount": "loan_amnt",
        "funded_amount": "funded_amnt",
        "interest_rate": "int_rate",
        "annual_income": "annual_inc",
        "employment_length": "emp_length",
        "debt_to_income": "dti",
        "status": "loan_status",
        # ... add your mappings
    })

    # Use it on every DB row
    db_row = get_customer_from_db(customer_id)
    mapped = mapper.map(db_row)
    result = get_recommendation(mapped)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Default mapping from common alternative names to Lending Club column names
DEFAULT_ALIASES = {
    # Loan details
    "loan_amount": "loan_amnt",
    "loanamount": "loan_amnt",
    "loan_amt": "loan_amnt",
    "principal": "loan_amnt",
    "funded_amount": "funded_amnt",
    "funded_amt": "funded_amnt",
    "interest_rate": "int_rate",
    "interestrate": "int_rate",
    "rate": "int_rate",
    "emi": "installment",
    "emi_amount": "installment",
    "monthly_installment": "installment",
    "loan_purpose": "purpose",
    "loan_grade": "grade",
    "risk_grade": "grade",
    "loan_term": "term",
    "tenure": "term",

    # Borrower profile
    "annual_income": "annual_inc",
    "annualincome": "annual_inc",
    "income": "annual_inc",
    "yearly_income": "annual_inc",
    "salary": "annual_inc",
    "employment_length": "emp_length",
    "emp_years": "emp_length",
    "experience": "emp_length",
    "work_experience": "emp_length",
    "home_ownership_status": "home_ownership",
    "housing": "home_ownership",
    "debt_to_income": "dti",
    "dti_ratio": "dti",
    "state": "addr_state",
    "address_state": "addr_state",
    "borrower_state": "addr_state",
    "app_type": "application_type",

    # Delinquency
    "status": "loan_status",
    "account_status": "loan_status",
    "current_status": "loan_status",
    "default_status": "loan_status",
    "dpd": "loan_status",  # If someone passes DPD, map to loan_status
    "days_past_due": "loan_status",
    "delinquencies_2yrs": "delinq_2yrs",
    "delinquency_2years": "delinq_2yrs",
    "current_delinquent": "acc_now_delinq",
    "delinquent_accounts": "acc_now_delinq",
    "delinquent_amount": "delinq_amnt",
    "months_since_delinquency": "mths_since_last_delinq",

    # Credit profile
    "revolving_balance": "revol_bal",
    "revolving_utilization": "revol_util",
    "bankcard_utilization": "bc_util",
    "open_accounts": "open_acc",
    "total_accounts": "total_acc",
    "public_records": "pub_rec",
    "bankruptcies": "pub_rec_bankruptcies",
    "inquiries_6months": "inq_last_6mths",
    "mortgage_accounts": "mort_acc",

    # Recovery
    "recovery_amount": "recoveries",
    "recovered": "recoveries",
    "collection_fee": "collection_recovery_fee",
}

# Valid Lending Club column names (the target namespace)
VALID_LC_COLUMNS = {
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "purpose", "application_type", "grade", "sub_grade",
    "annual_inc", "emp_length", "home_ownership", "verification_status",
    "addr_state", "dti", "earliest_cr_line",
    "pub_rec", "pub_rec_bankruptcies", "tax_liens",
    "acc_now_delinq", "delinq_2yrs", "delinq_amnt",
    "mths_since_last_delinq", "collections_12_mths_ex_med",
    "chargeoff_within_12_mths", "num_accts_ever_120_pd",
    "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "revol_bal", "revol_util", "bc_util",
    "avg_cur_bal", "tot_cur_bal", "total_bal_ex_mort",
    "total_bc_limit", "percent_bc_gt_75",
    "open_acc", "total_acc", "mort_acc",
    "pct_tl_nvr_dlq", "inq_last_6mths", "acc_open_past_24mths",
    "loan_status", "recoveries", "collection_recovery_fee",
}


class ColumnMapper:
    """
    Maps external database column names to Lending Club column names.

    The mapper auto-detects common aliases (e.g., "loan_amount" -> "loan_amnt")
    and allows custom mappings. Columns that already use LC names pass through.

    Usage:
        # Option 1: Use default aliases (auto-detects common names)
        mapper = ColumnMapper()
        mapped = mapper.map(db_row_dict)

        # Option 2: Custom mapping for your specific DB schema
        mapper = ColumnMapper(custom_mapping={
            "your_db_column": "lc_column_name",
        })

        # Option 3: Combine both
        mapper = ColumnMapper(custom_mapping={"custom_col": "loan_amnt"})
    """

    def __init__(self, custom_mapping: Optional[dict[str, str]] = None):
        """
        Args:
            custom_mapping: Dict mapping your DB column names to LC column names.
                Merged with (and overrides) the default aliases.
        """
        # Start with default aliases, then overlay custom mappings
        self._mapping = dict(DEFAULT_ALIASES)
        if custom_mapping:
            self._mapping.update(custom_mapping)

    def map(self, row: dict) -> dict:
        """
        Map a database row dict to Lending Club column names.

        - If a key is already a valid LC column name, it passes through.
        - If a key matches a known alias, it's renamed.
        - Unknown keys are passed through unchanged (the sanitizer will handle PII).

        Args:
            row: Dict from your database query

        Returns:
            Dict with keys mapped to LC column names
        """
        mapped = {}
        unmapped = []

        for key, value in row.items():
            key_lower = key.lower().strip()

            if key_lower in VALID_LC_COLUMNS:
                # Already a valid LC column name
                mapped[key_lower] = value
            elif key_lower in self._mapping:
                # Known alias -> map to LC name
                lc_name = self._mapping[key_lower]
                mapped[lc_name] = value
            else:
                # Unknown column -> pass through (sanitizer handles PII)
                mapped[key] = value
                unmapped.append(key)

        if unmapped:
            logger.debug(f"Unmapped columns passed through: {unmapped}")

        return mapped

    def validate(self, row: dict) -> dict:
        """
        Map and validate a row, returning info about coverage.

        Returns:
            Dict with mapped data and validation info
        """
        mapped = self.map(row)

        # Check which critical fields are present
        critical_fields = [
            "loan_amnt", "loan_status", "grade", "annual_inc", "dti"
        ]
        missing_critical = [f for f in critical_fields if f not in mapped]

        important_fields = [
            "int_rate", "term", "purpose", "home_ownership",
            "delinq_2yrs", "acc_now_delinq",
        ]
        missing_important = [f for f in important_fields if f not in mapped]

        return {
            "mapped_data": mapped,
            "total_fields": len(mapped),
            "missing_critical": missing_critical,
            "missing_important": missing_important,
            "ready": len(missing_critical) == 0,
            "warning": (
                f"Missing critical fields: {missing_critical}"
                if missing_critical
                else None
            ),
        }
