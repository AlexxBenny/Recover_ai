"""
Generate sample bank policy PDFs for testing the RAG engine.

Run this once to create the policy PDFs in the policies/ directory:
    python scripts/create_sample_policies.py

These simulate realistic Indian bank/NBFC loan recovery policies.
"""

from pathlib import Path

from fpdf import FPDF

POLICIES_DIR = Path(__file__).resolve().parent.parent / "policies"


def create_pdf(filename: str, title: str, content: str) -> Path:
    """Create a PDF from text content."""
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = POLICIES_DIR / filename

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 12, _sanitize(title), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Sanitize all content at once
    content = _sanitize(content)

    # Body — process line by line for basic formatting
    for line in content.strip().split("\n"):
        stripped = line.strip()

        if stripped.startswith("## "):
            pdf.ln(4)
            pdf.set_font("Helvetica", style="B", size=13)
            pdf.cell(0, 8, stripped[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif stripped.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.cell(0, 7, stripped[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif stripped.startswith("---"):
            pdf.ln(3)
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 170, pdf.get_y())
            pdf.ln(3)
        elif stripped == "":
            pdf.ln(3)
        else:
            pdf.set_font("Helvetica", size=10)
            pdf.set_x(pdf.l_margin)  # Ensure cursor is at left margin
            try:
                pdf.multi_cell(0, 5, stripped)
            except Exception as e:
                print(f"  ERROR on line: '{stripped[:80]}...'")
                print(f"  X={pdf.get_x()}, margin={pdf.l_margin}, page_w={pdf.w}")
                raise

    pdf.output(str(filepath))
    return filepath


def _sanitize(text: str) -> str:
    """Replace characters that Helvetica/latin-1 can't render."""
    # Common Unicode replacements
    replacements = {
        "\u20b9": "Rs ",  # ₹
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2022": "-",    # bullet
        "\u2026": "...",  # ellipsis
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Encode to latin-1, replacing anything still unsupported
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ===========================================================================
# Policy 1: Collection Strategy Policy
# ===========================================================================

COLLECTION_STRATEGY = """
## 1. Purpose and Scope

This policy defines the collection strategy framework for recovering overdue loan amounts from defaulting borrowers. It applies to all retail lending products including personal loans, credit cards, auto loans, and small business loans.

The collection strategy is determined based on the borrower's Days Past Due (DPD), outstanding loan amount, risk grade, and repayment history.

---

## 2. DPD Bucket Classification

Borrowers are classified into the following DPD buckets:

### Bucket X: DPD 0 (Current)
No collection action required. Standard servicing and relationship management.

### Bucket 1: DPD 1-30 (Early Delinquency)
Priority: Low
Primary Action: Automated reminders only
- Day 1-3: Send automated SMS reminder about missed payment
- Day 4-7: Send email reminder with payment link and amount details
- Day 7-14: Send IVR (Interactive Voice Response) automated call
- Day 15-30: Assign to tele-calling team for a single courtesy call
- No demand notices at this stage
- Tone: Polite and supportive. Treat as a potential oversight.

For loan amounts below Rs 1,00,000: Only automated SMS and email. No phone calls.
For loan amounts Rs 1,00,000 to Rs 5,00,000: SMS, email, and one courtesy call after Day 15.
For loan amounts above Rs 5,00,000: SMS, email, and courtesy call after Day 10.

### Bucket 2: DPD 31-60 (Moderate Delinquency)
Priority: Medium
Primary Action: Active tele-calling and written communication
- Day 31-35: Assign dedicated tele-calling agent
- Day 31-45: Make 2-3 phone calls per week (within permitted hours)
- Day 35-40: Send formal written reminder letter via email and post
- Day 45-60: Send first demand notice (soft notice) via registered post
- Offer EMI restructuring for borrowers showing genuine financial difficulty
- Tone: Firm but professional. Emphasize consequences of continued default.

For loan amounts below Rs 1,00,000: Tele-calling only, no physical letters.
For loan amounts Rs 1,00,000 to Rs 5,00,000: Tele-calling + email + soft demand notice.
For loan amounts Rs 5,00,000 to Rs 25,00,000: Intensive tele-calling + demand notice + consider field visit.
For loan amounts above Rs 25,00,000: Assign senior recovery officer. Mandatory demand notice.

### Bucket 3: DPD 61-90 (Serious Delinquency)
Priority: High
Primary Action: Intensive collection with escalation
- Day 61-70: Escalate to hard collection team
- Day 61-75: Daily phone call attempts (within permitted hours)
- Day 65-75: Send second demand notice (firm notice) with legal warning
- Day 75-90: Initiate field visit by collection agent to borrower's address
- Day 80-90: Send legal notice under Section 138 NI Act (if cheque bounce)
- Offer One Time Settlement (OTS) with up to 10% discount on total outstanding
- Tone: Serious and formal. Reference legal consequences explicitly.

For loan amounts below Rs 1,00,000: Intensive tele-calling + firm demand notice.
For loan amounts Rs 1,00,000 to Rs 5,00,000: Add field visit after Day 75.
For loan amounts above Rs 5,00,000: Mandatory field visit by Day 70. Legal notice preparation.

### Bucket 4: DPD 91-180 (NPA Stage)
Priority: Critical
Primary Action: Legal action and recovery proceedings
- Day 91: Account classified as Non-Performing Asset (NPA) per RBI guidelines
- Day 91-100: Send final demand notice with 15-day legal action timeline
- Day 100-120: Initiate legal proceedings (filing suit for recovery)
- Day 91-180: Weekly field visits by senior recovery agent
- Day 120-150: For secured loans, initiate SARFAESI proceedings
- Offer OTS with up to 20% discount on total outstanding
- Assign to external legal counsel for high-value accounts
- Tone: Strictly legal and formal.

For loan amounts below Rs 1,00,000: External collection agency referral.
For loan amounts Rs 1,00,000 to Rs 25,00,000: In-house legal + field recovery.
For loan amounts above Rs 25,00,000: External legal counsel + SARFAESI (if secured).

### Bucket 5: DPD 180+ (Write-off Candidate)
Priority: Critical
Primary Action: Recovery through legal and settlement channels
- Day 180-270: Continue legal proceedings
- Day 270+: Evaluate for write-off or sale to Asset Reconstruction Company (ARC)
- Offer OTS with up to 40% discount on total outstanding for early closure
- For secured loans: Proceed with asset seizure and auction under SARFAESI
- Maintain contact attempts quarterly
- Report to credit bureaus (CIBIL, Experian, etc.)

---

## 3. Risk Grade Based Adjustments

### Grade A-B (Low Risk)
- These borrowers typically have strong credit histories
- Use softer collection approach, extend timelines by 7-10 days before escalation
- Prioritize restructuring and settlement offers
- Higher probability of recovery through negotiation

### Grade C-D (Medium Risk)
- Follow standard collection timelines as defined above
- Monitor closely for payment promises and follow up

### Grade E-F (High Risk)
- Accelerate collection timelines by 5-7 days
- Earlier assignment to field collection teams
- Proactive legal notice preparation
- Lower OTS discount thresholds

### Grade G (Very High Risk)
- Immediately assign to senior recovery officer at DPD 31+
- Fast-track legal proceedings at DPD 61+
- Mandatory field visits starting DPD 45
- Consider external collection agency referral early

---

## 4. Channel Priority Matrix

For each DPD stage, use the following channel priority:

DPD 1-30: SMS > Email > IVR > Phone Call
DPD 31-60: Phone Call > Email > SMS > Physical Letter
DPD 61-90: Phone Call > Field Visit > Legal Notice > Email
DPD 91-180: Field Visit > Legal Action > Phone Call > Email
DPD 180+: Legal Action > Field Visit > Settlement Offer
"""

# ===========================================================================
# Policy 2: Escalation Matrix
# ===========================================================================

ESCALATION_MATRIX = """
## 1. Purpose

This document defines the escalation framework for loan collection activities. It specifies when and how cases should be escalated from one collection stage to the next, and which teams are responsible at each level.

---

## 2. Collection Teams and Responsibilities

### Level 1: Soft Collection Team (Tele-calling)
- Handles: DPD 1-30 accounts
- Team Size: Based on portfolio size
- Target: Resolve 70% of cases before DPD 30
- Reporting: Daily call logs, weekly resolution reports
- KPI: Promise-to-Pay (PTP) conversion rate, right-party contact rate

### Level 2: Intensive Collection Team
- Handles: DPD 31-60 accounts, unresolved Level 1 cases
- Actions: Dedicated agent assignment, demand notices, restructuring offers
- Target: Resolve 50% of cases before DPD 60
- Reporting: Weekly case reviews with team lead
- KPI: Collection efficiency, EMI restructuring success rate

### Level 3: Hard Collection and Field Team
- Handles: DPD 61-90 accounts, unresolved Level 2 cases
- Actions: Field visits, legal notice preparation, OTS negotiations
- Target: Initiate contact with 90% of accounts through field visits
- Reporting: Field visit reports with GPS tracking
- KPI: Field contact rate, recovery amount per visit

### Level 4: Legal and Recovery Team
- Handles: DPD 90+ accounts, NPA classified accounts
- Actions: Legal proceedings, SARFAESI, asset seizure, ARC referral
- Target: Maximize recovery through legal channels
- Reporting: Monthly legal proceedings status
- KPI: Legal recovery rate, cost of recovery

---

## 3. Escalation Triggers

### Automatic Escalation Criteria

From Level 1 to Level 2:
- DPD crosses 30 days
- Borrower unreachable after 5 contact attempts
- Borrower refuses to pay or disputes the debt
- Cheque bounce or auto-debit failure more than twice

From Level 2 to Level 3:
- DPD crosses 60 days
- No payment received after demand notice
- Borrower breaks Promise-to-Pay (PTP) twice
- Loan amount exceeds Rs 5,00,000 and DPD exceeds 45 days

From Level 3 to Level 4:
- DPD crosses 90 days (NPA classification)
- Borrower untraceable after field visits
- Total outstanding exceeds Rs 10,00,000
- Suspected fraud or willful default

### Priority Escalation (Fast-Track)

The following conditions trigger immediate escalation regardless of DPD:
- Loan amount above Rs 50,00,000: Assign Level 3 team at DPD 31
- Grade G with loan above Rs 10,00,000: Assign Level 3 team at DPD 31
- Borrower threatens or is abusive: Escalate to legal team immediately
- Suspected fraud indicators: Escalate to fraud and legal team immediately
- Borrower files bankruptcy or insolvency: Immediate legal team involvement

---

## 4. Manager Escalation Thresholds

### Team Lead Review Required
- Any account with outstanding above Rs 25,00,000
- Cases where borrower has filed formal complaints
- Cases with more than 3 broken PTP commitments
- Accounts showing unusual patterns (partial payments, frequent disputes)

### Regional Manager Review Required
- Any account with outstanding above Rs 1,00,00,000
- Cases involving politically exposed persons (PEPs)
- Cases requiring SARFAESI proceedings
- Group defaults (multiple loans from same borrower/guarantor)

### Head of Collections Review Required
- Write-off recommendations above Rs 50,00,000
- Settlement offers with discount above 30%
- Cases involving legal counter-suits by borrowers
- Media-sensitive cases

---

## 5. Escalation Timelines

| From | To | Maximum Time |
|------|------|-------------|
| Level 1 | Level 2 | DPD 30 or 5 failed contacts |
| Level 2 | Level 3 | DPD 60 or 2 broken PTPs |
| Level 3 | Level 4 | DPD 90 or borrower untraceable |
| Level 4 | External Legal | DPD 120 or suit filing required |
| Level 4 | ARC Referral | DPD 270 or write-off evaluation |

---

## 6. De-escalation Criteria

An account can be de-escalated to a lower level when:
- Borrower makes a partial payment covering at least 50% of overdue amount
- Borrower enters into a formal EMI restructuring agreement
- DPD reduces to below the threshold of the current level
- Settlement agreement is signed and first installment received
"""

# ===========================================================================
# Policy 3: Communication Guidelines
# ===========================================================================

COMMUNICATION_GUIDELINES = """
## 1. Purpose

This document provides guidelines for all communications with defaulting borrowers, covering tone, channel selection, message content, and special situations like settlement and restructuring offers.

---

## 2. Communication Tone by DPD Stage

### DPD 1-30: Supportive and Courteous
- Address the borrower by name
- Assume the missed payment was an oversight
- Provide clear payment instructions and amount due
- Offer to help with any payment difficulties
- Sample tone: "Dear [Name], we noticed your EMI of Rs [amount] due on [date] has not been received. This may be an oversight. Please make the payment at your earliest convenience."

### DPD 31-60: Professional and Firm
- Clearly state the overdue amount and duration
- Mention the consequences of continued default (late fees, credit score impact)
- Offer restructuring options if the borrower expresses financial difficulty
- Sample tone: "Dear [Name], your account is overdue by [X] days with an outstanding of Rs [amount]. Continued non-payment will affect your credit score and may attract additional charges. Please contact us to discuss resolution options."

### DPD 61-90: Serious and Formal
- Reference specific policy sections and regulatory requirements
- Explicitly mention legal consequences
- Provide a clear deadline for resolution
- Present settlement options where eligible
- Sample tone: "Dear [Name], despite our previous communications, your account remains overdue by [X] days. We are obligated to escalate this matter. Please settle the outstanding amount of Rs [amount] within 15 days to avoid further action."

### DPD 90+: Legal and Final
- Use formal legal language
- Reference specific acts and regulations
- Provide final deadlines with explicit consequences
- All communication to be reviewed by legal team
- Sample tone: "FINAL NOTICE: Your loan account [number] with outstanding of Rs [amount] is classified as Non-Performing. Legal proceedings will be initiated if payment is not received within 15 days of this notice."

---

## 3. Settlement and Restructuring Criteria

### EMI Restructuring Eligibility
Borrowers may be offered EMI restructuring if:
- They demonstrate genuine financial hardship (job loss, medical emergency, business downturn)
- They have made at least 6 EMI payments before default
- Current DPD is between 31 and 90 days
- They can provide documentation of financial difficulty
- Restructuring options include: tenure extension, reduced EMI, moratorium period (up to 3 months)

### One Time Settlement (OTS) Eligibility

OTS Discount Schedule:
- DPD 61-90: Up to 10% discount on total outstanding (principal + interest + charges)
- DPD 91-180: Up to 20% discount on total outstanding
- DPD 181-365: Up to 30% discount on total outstanding
- DPD 365+: Up to 40% discount on total outstanding (requires Head of Collections approval)

OTS Conditions:
- Settlement amount must be paid in lump sum or maximum 3 installments within 90 days
- Borrower must sign a full and final settlement agreement
- No further claims can be made by either party after settlement
- Credit bureau reporting will be updated to "Settled" (not "Closed")

### Settlement Authority Matrix
- Up to Rs 5,00,000 outstanding: Team Lead can approve
- Rs 5,00,000 to Rs 25,00,000: Regional Manager approval required
- Rs 25,00,000 to Rs 1,00,00,000: Head of Collections approval required
- Above Rs 1,00,00,000: Board-level approval required

---

## 4. Channel Selection Guidelines

### SMS
- Use for: Payment reminders (DPD 1-30), payment confirmation, EMI due alerts
- Frequency: Maximum 2 SMS per week
- Timing: Between 9 AM and 7 PM only
- Content: Brief, include amount and payment link

### Email
- Use for: Detailed communication, demand notices, restructuring offers, settlement proposals
- Frequency: Maximum 3 emails per week
- Include: Full account details, payment options, contact information
- Attach: Relevant documents (account statement, demand notice PDF)

### Phone Call
- Use for: DPD 15+ accounts, follow-up on payment promises, restructuring discussions
- Frequency: Maximum 1 call per day, 3 calls per week
- Timing: Between 8 AM and 7 PM only (as per RBI guidelines)
- Duration: Keep under 5 minutes unless borrower is engaged in resolution discussion
- Recording: All calls must be recorded

### Physical Letter / Registered Post
- Use for: Demand notices (DPD 45+), legal notices (DPD 80+), final notices (DPD 91+)
- Must include: Full account details, amount due, deadline, consequences, grievance contact
- Registered post for all legal and final notices

### Field Visit
- Use for: DPD 60+ accounts, high-value accounts, untraceable borrowers
- Must: Carry proper identification and authorization letter
- Must: Maintain dignified and professional conduct
- Must: File detailed visit report within 24 hours
- Must Not: Visit before 8 AM or after 7 PM
- Must Not: Use force, threats, or any form of intimidation

---

## 5. Prohibited Communications

The following are strictly prohibited in all borrower communications:
- Threatening language or intimidation of any kind
- Contacting borrower's family, friends, or employer about the debt (unless they are guarantors)
- Sending messages that could be embarrassing if seen by others
- Using abusive, aggressive, or discriminatory language
- Disclosing the borrower's default status to unauthorized third parties
- Making false claims about legal proceedings or consequences
- Calling outside permitted hours (before 8 AM or after 7 PM)
- Excessive contact (more than 3 calls per day or 5 contacts per day across all channels)
"""

# ===========================================================================
# Policy 4: Regulatory Compliance
# ===========================================================================

REGULATORY_COMPLIANCE = """
## 1. Purpose

This document outlines the regulatory framework governing loan collection and recovery activities. All collection agents and teams must comply with these guidelines. Violations may result in regulatory penalties and disciplinary action.

---

## 2. RBI Fair Practices Code for Debt Collection

The Reserve Bank of India (RBI) has issued comprehensive guidelines on fair practices in debt collection. Key requirements include:

### Contact Rules
- Collection calls may only be made between 8:00 AM and 7:00 PM
- The borrower must be contacted at a reasonable frequency — no harassment
- All communication must clearly identify the lender and the agent
- Collection agents must carry proper identification during field visits
- Calls must be recorded and records maintained for a minimum of 2 years

### Prohibited Practices
- Use of threatening, abusive, or obscene language
- Calling on known holidays or festivals (unless pre-agreed by borrower)
- Contacting borrower at their workplace unless specifically permitted
- Disclosing debt information to unauthorized third parties
- Contacting co-applicants/guarantors before exhausting contact with primary borrower
- Using physical force or criminal intimidation
- Damaging borrower's property or reputation
- Publishing names of defaulters publicly (name-and-shame)

### Borrower Rights
- Right to receive clear information about the outstanding amount and calculation
- Right to request communication in their preferred language (Hindi/English/regional)
- Right to request a statement of account at any time
- Right to file a complaint through the lender's grievance redressal mechanism
- Right to approach the Banking Ombudsman if the complaint is not resolved in 30 days
- Right to request cessation of contact during specific hours or days (with reasonable limits)
- Right to dispute the debt amount and receive a response within 14 days

---

## 3. SARFAESI Act (Securitisation and Reconstruction of Financial Assets)

The SARFAESI Act, 2002 applies to secured loans where the outstanding is Rs 1,00,000 or above.

### Applicability
- Only for secured loans (backed by collateral)
- Outstanding amount must be Rs 1,00,000 or more
- Account must be classified as NPA (DPD 90+)
- Does not apply to agricultural land

### Procedure
Step 1: Issue a demand notice under Section 13(2) giving 60 days to repay
Step 2: If no payment received, take possession of the secured asset under Section 13(4)
Step 3: Issue public notice for auction/sale of the asset
Step 4: Conduct auction with a reserve price (not less than assessed value)
Step 5: Apply sale proceeds to the outstanding loan amount
Step 6: Return any surplus to the borrower

### Borrower's Right Under SARFAESI
- The borrower can make a representation within 60 days of the demand notice
- The borrower can file an application with the Debt Recovery Tribunal (DRT)
- The lender must respond to the representation within 15 days

---

## 4. Lok Adalat and Debt Recovery Tribunal (DRT)

### Lok Adalat
- For cases with outstanding up to Rs 20,00,000
- Encourages negotiated settlement between lender and borrower
- Decisions are binding and non-appealable
- No court fee required
- Typically faster resolution (1-3 months)

### Debt Recovery Tribunal (DRT)
- For cases with outstanding above Rs 20,00,000
- Formal legal proceedings with evidence and arguments
- Decision can be appealed to Debt Recovery Appellate Tribunal (DRAT)
- Typical resolution timeline: 6-18 months
- Court fee applicable based on claim amount

---

## 5. Data Privacy and Confidentiality

### Credit Bureau Reporting
- Delinquency must be reported to all credit bureaus (CIBIL, Experian, Equifax, CRIF High Mark)
- Reporting frequency: Monthly
- Reporting must be accurate and timely
- Any dispute raised by borrower must be investigated and resolved within 30 days
- After settlement/closure, bureau records must be updated within 30 days

### Data Protection
- Borrower's personal and financial data must be treated as strictly confidential
- Data sharing with third parties (collection agencies) requires borrower consent or contract
- All digital communications must be encrypted
- Physical documents must be stored securely and destroyed as per retention policy
- Collection agents must not store borrower data on personal devices

---

## 6. External Collection Agency Guidelines

When outsourcing collection to external agencies:
- Agency must be empanelled and approved by the bank's board
- Agency staff must be trained on RBI fair practices code
- Agency must maintain call recordings and visit logs
- Bank retains responsibility for the agency's conduct
- Regular audits of agency practices must be conducted quarterly
- Complaints against agency must be handled by the bank's grievance mechanism
- Agency fees: Typically 5-15% of recovered amount based on DPD bucket and loan type

---

## 7. Documentation Requirements

All collection activities must be documented:
- Phone calls: Date, time, duration, summary, outcome, recording reference
- SMS/Email: Timestamp, content, delivery status
- Letters: Date sent, mode of dispatch, tracking number
- Field visits: Date, time, location, agent name, borrower response, photos (if applicable)
- Legal notices: Date, content, dispatch mode, acknowledgment receipt
- Settlement discussions: Terms proposed, borrower response, approval chain
- All documents must be retained for a minimum of 8 years after account closure
"""


def main():
    """Generate all sample policy PDFs."""
    policies = [
        (
            "collection_strategy_policy.pdf",
            "Collection Strategy Policy",
            COLLECTION_STRATEGY,
        ),
        (
            "escalation_matrix.pdf",
            "Escalation Matrix",
            ESCALATION_MATRIX,
        ),
        (
            "communication_guidelines.pdf",
            "Communication Guidelines",
            COMMUNICATION_GUIDELINES,
        ),
        (
            "regulatory_compliance.pdf",
            "Regulatory Compliance Guidelines",
            REGULATORY_COMPLIANCE,
        ),
    ]

    print(f"Generating policy PDFs in: {POLICIES_DIR}\n")

    for filename, title, content in policies:
        filepath = create_pdf(filename, title, content)
        print(f"  Created: {filepath.name}")

    print(f"\nDone! {len(policies)} policy PDFs created.")


if __name__ == "__main__":
    main()
