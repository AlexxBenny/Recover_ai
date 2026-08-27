"""
Prompt templates for the RAG engine.

Contains:
- System prompt defining the LLM's role
- Query template for formatting customer details + retrieved context
- Helper functions for building retrieval queries
- Loan status to DPD mapping
"""

# ---------------------------------------------------------------------------
# Loan status → DPD bucket mapping
# ---------------------------------------------------------------------------

# Maps Lending Club loan_status values to approximate DPD ranges
LOAN_STATUS_DPD_MAP = {
    "Current": 0,
    "In Grace Period": 15,
    "Late (16-30 days)": 30,
    "Late (31-120 days)": 75,
    "Default": 150,
    "Charged Off": 180,
}


def estimate_dpd(loan_status: str) -> int:
    """
    Estimate Days Past Due from Lending Club loan_status.

    Args:
        loan_status: One of 'Current', 'In Grace Period',
            'Late (16-30 days)', 'Late (31-120 days)', 'Default', 'Charged Off'

    Returns:
        Approximate DPD as an integer
    """
    return LOAN_STATUS_DPD_MAP.get(loan_status, 0)


def get_dpd_bucket(loan_status: str) -> str:
    """Return a human-readable DPD bucket label from loan_status."""
    buckets = {
        "Current": "DPD 0 (Current)",
        "In Grace Period": "DPD 1-15 (Grace Period)",
        "Late (16-30 days)": "DPD 16-30 (Early Delinquency)",
        "Late (31-120 days)": "DPD 31-120 (Moderate to Serious Delinquency)",
        "Default": "DPD 120+ (Default / NPA)",
        "Charged Off": "DPD 180+ (Charged Off / Write-off)",
    }
    return buckets.get(loan_status, f"Unknown ({loan_status})")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior collections strategy advisor for a bank's loan recovery department.
Your role is to analyze defaulter profiles and recommend the most appropriate collection
strategy based on the bank's internal policies.

You have access to the bank's collection policies, escalation matrix, communication
guidelines, and regulatory compliance documents. Use these to make informed,
policy-compliant recommendations.

When providing recommendations, you must:
1. Summarize the customer's current default status
2. Identify the applicable DPD (Days Past Due) bucket
3. Recommend specific collection actions based on policy
4. Explain WHY this strategy is appropriate for this specific customer
5. Flag any regulatory constraints that apply
6. Suggest the priority level (Low / Medium / High / Critical)

Always be factual and reference specific policy guidelines in your reasoning.
"""


# ---------------------------------------------------------------------------
# Query template — aligned with Lending Club dataset columns
# ---------------------------------------------------------------------------

QUERY_TEMPLATE = """\
Based on the bank's collection policies, recommend the appropriate recovery strategy
for the following defaulter. All monetary values are in Indian Rupees (Rs / INR).

**Loan Details:**
- Loan Amount: {loan_amnt}
- Funded Amount: {funded_amnt}
- Term: {term}
- Interest Rate: {int_rate}%
- Monthly Installment: {installment}
- Loan Purpose: {purpose}
- Loan Grade: {grade} (Sub-grade: {sub_grade})

**Borrower Profile:**
- Annual Income: {annual_inc}
- Employment Length: {emp_length}
- Home Ownership: {home_ownership}
- Debt-to-Income Ratio (DTI): {dti}
- Application Type: {application_type}
- State: {addr_state}

**Delinquency Status:**
- Current Loan Status: {loan_status}
- Estimated DPD Bucket: {dpd_bucket}
- Estimated Days Past Due: {estimated_dpd}
- Currently Delinquent Accounts: {acc_now_delinq}
- Delinquencies in Last 2 Years: {delinq_2yrs}
- Delinquent Amount: {delinq_amnt}
- Months Since Last Delinquency: {mths_since_last_delinq}
- Accounts 30 Days Past Due: {num_tl_30dpd}
- Accounts 90+ Days Past Due (last 24m): {num_tl_90g_dpd_24m}
- Accounts Ever 120 Days Past Due: {num_accts_ever_120_pd}

**Credit Profile:**
- Revolving Balance: {revol_bal}
- Revolving Utilization: {revol_util}%
- Bankcard Utilization: {bc_util}%
- Open Accounts: {open_acc}
- Total Accounts: {total_acc}
- Public Records: {pub_rec}
- Bankruptcies: {pub_rec_bankruptcies}
- Tax Liens: {tax_liens}
- Inquiries Last 6 Months: {inq_last_6mths}
- Percent Never Delinquent: {pct_tl_nvr_dlq}%

**Recovery Info:**
- Recoveries to Date: {recoveries}
- Collection Recovery Fee: {collection_recovery_fee}

**Relevant Bank Policies:**
{context}

**Instructions:**
Provide your recommendation in the following structure:

1. **Customer Status Summary**: Brief overview of the defaulter's situation and financial profile
2. **DPD Bucket**: Which DPD category this falls under per bank policy
3. **Risk Level**: Low / Medium / High / Critical
4. **Recommended Strategy**: Specific actions to take (e.g., SMS, call, notice, field visit)
5. **Communication Channel**: Primary and secondary channels to use
6. **Escalation Required**: Yes/No — and to which level/team
7. **Settlement Eligibility**: Whether OTS/restructuring should be offered, with terms
8. **Reasoning**: Why this strategy was chosen, referencing specific policy sections
9. **Regulatory Notes**: Any compliance constraints to be aware of
"""


def format_customer_query(customer: dict, context: str) -> str:
    """
    Format customer details and retrieved context into the full LLM prompt.

    Accepts a dict with Lending Club dataset column names.

    Args:
        customer: Dict with customer details using Lending Club column names
        context: Retrieved policy text chunks joined together

    Returns:
        Formatted prompt string ready for the LLM
    """
    # Derive DPD info from loan_status
    loan_status = customer.get("loan_status", "Unknown")
    estimated_dpd = estimate_dpd(loan_status)
    dpd_bucket = get_dpd_bucket(loan_status)

    return QUERY_TEMPLATE.format(
        loan_amnt=customer.get("loan_amnt", "N/A"),
        funded_amnt=customer.get("funded_amnt", "N/A"),
        term=customer.get("term", "N/A"),
        int_rate=customer.get("int_rate", "N/A"),
        installment=customer.get("installment", "N/A"),
        purpose=customer.get("purpose", "N/A"),
        grade=customer.get("grade", "N/A"),
        sub_grade=customer.get("sub_grade", "N/A"),
        annual_inc=customer.get("annual_inc", "N/A"),
        emp_length=customer.get("emp_length", "N/A"),
        home_ownership=customer.get("home_ownership", "N/A"),
        dti=customer.get("dti", "N/A"),
        application_type=customer.get("application_type", "N/A"),
        addr_state=customer.get("addr_state", "N/A"),
        loan_status=loan_status,
        dpd_bucket=dpd_bucket,
        estimated_dpd=estimated_dpd,
        acc_now_delinq=customer.get("acc_now_delinq", "N/A"),
        delinq_2yrs=customer.get("delinq_2yrs", "N/A"),
        delinq_amnt=customer.get("delinq_amnt", "N/A"),
        mths_since_last_delinq=customer.get("mths_since_last_delinq", "N/A"),
        num_tl_30dpd=customer.get("num_tl_30dpd", "N/A"),
        num_tl_90g_dpd_24m=customer.get("num_tl_90g_dpd_24m", "N/A"),
        num_accts_ever_120_pd=customer.get("num_accts_ever_120_pd", "N/A"),
        revol_bal=customer.get("revol_bal", "N/A"),
        revol_util=customer.get("revol_util", "N/A"),
        bc_util=customer.get("bc_util", "N/A"),
        open_acc=customer.get("open_acc", "N/A"),
        total_acc=customer.get("total_acc", "N/A"),
        pub_rec=customer.get("pub_rec", "N/A"),
        pub_rec_bankruptcies=customer.get("pub_rec_bankruptcies", "N/A"),
        tax_liens=customer.get("tax_liens", "N/A"),
        inq_last_6mths=customer.get("inq_last_6mths", "N/A"),
        pct_tl_nvr_dlq=customer.get("pct_tl_nvr_dlq", "N/A"),
        recoveries=customer.get("recoveries", "N/A"),
        collection_recovery_fee=customer.get("collection_recovery_fee", "N/A"),
        context=context,
    )


def build_retrieval_query(customer: dict) -> str:
    """
    Build a focused query string for vector store similarity search.

    Instead of sending raw customer data, this constructs a semantically
    meaningful query that will match relevant policy sections.

    Args:
        customer: Dict with customer details using Lending Club column names

    Returns:
        A natural-language query string optimized for retrieval
    """
    loan_status = customer.get("loan_status", "")
    loan_amnt = customer.get("loan_amnt", 0)
    grade = customer.get("grade", "")
    estimated_dpd = estimate_dpd(loan_status)

    query_parts = [
        f"Collection strategy for loan status {loan_status}",
        f"approximately {estimated_dpd} days past due",
        f"loan amount {loan_amnt}",
        f"risk grade {grade}",
    ]

    # Add DPD-specific keywords to improve retrieval precision
    if estimated_dpd > 180:
        query_parts.append("NPA write-off legal recovery SARFAESI charged off")
    elif estimated_dpd > 90:
        query_parts.append("NPA classification legal action demand notice field visit default")
    elif estimated_dpd > 60:
        query_parts.append("hard collection demand notice intensive follow-up")
    elif estimated_dpd > 30:
        query_parts.append("phone call follow-up escalation reminder late payment")
    elif estimated_dpd > 0:
        query_parts.append("soft collection SMS email automated reminder grace period early stage")
    else:
        query_parts.append("current account monitoring")

    # Add loan amount context
    if loan_amnt > 25000:
        query_parts.append("high value loan priority")
    elif loan_amnt < 5000:
        query_parts.append("small ticket loan")

    # Add delinquency history context
    delinq_2yrs = customer.get("delinq_2yrs", 0)
    if delinq_2yrs and delinq_2yrs > 2:
        query_parts.append("repeat defaulter history of delinquency")

    pub_rec = customer.get("pub_rec", 0)
    if pub_rec and pub_rec > 0:
        query_parts.append("public records bankruptcy legal history")

    return " ".join(query_parts)
