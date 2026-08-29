#!/usr/bin/env python3
"""
FastAPI REST API for CatBoost Net Recovery Prediction
=====================================================
Two-Stage Hurdle Model:
  Stage 1: CatBoost Classifier  → P(net_recovery > 0)
  Stage 2: CatBoost Regressor   → E(net_recovery | net_recovery > 0)
  Combined: predicted_recovery = P(Y > 0) × E(Y | Y > 0)

Endpoints:
  GET  /                → API info & health check
  GET  /health          → Health check
  GET  /model/metrics   → Model performance metrics
  POST /predict         → Single loan prediction
  POST /predict/batch   → Batch prediction (multiple loans)

Usage:
  uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier, CatBoostRegressor

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_DIR = '/home/ubuntu/case study/saved_models'
CLS_PATH  = os.path.join(MODEL_DIR, 'catboost_classifier.cbm')
REG_PATH  = os.path.join(MODEL_DIR, 'catboost_regressor_pos.cbm')
META_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')

# Global model references
classifier = None
regressor  = None
metadata   = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global classifier, regressor, metadata

    print("Loading CatBoost models...")
    classifier = CatBoostClassifier()
    classifier.load_model(CLS_PATH)

    regressor = CatBoostRegressor()
    regressor.load_model(REG_PATH)

    with open(META_PATH, 'r') as f:
        metadata = json.load(f)

    print(f"Models loaded. Features: {len(metadata['feature_cols'])}, "
          f"Categorical: {len(metadata['cat_cols'])}")
    yield
    print("Shutting down API server.")


# ─────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────
app = FastAPI(
    title="CatBoost Net Recovery Prediction API",
    description="Predict the net recovery amount a defaulted borrower will pay back. "
                "Uses a Two-Stage Hurdle CatBoost model trained on LendingClub data.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request/Response Schemas
# ─────────────────────────────────────────────
class LoanInput(BaseModel):
    """Input schema for a single loan prediction request."""
    loan_amnt: float = Field(..., description="Total loan amount ($)", example=15000)
    funded_amnt: float = Field(..., description="Funded amount ($)", example=15000)
    term: str = Field(..., description="Loan term", example="36 months")
    int_rate: float = Field(..., description="Interest rate (%)", example=13.56)
    installment: float = Field(..., description="Monthly installment ($)", example=509.47)
    purpose: str = Field(..., description="Loan purpose", example="debt_consolidation")
    application_type: str = Field("Individual", description="Individual or Joint App")
    grade: str = Field(..., description="LC grade (A-G)", example="C")
    sub_grade: str = Field(..., description="LC sub-grade", example="C3")
    annual_inc: float = Field(..., description="Annual income ($)", example=75000)
    emp_length: str = Field("10+ years", description="Employment length")
    home_ownership: str = Field(..., description="RENT, OWN, MORTGAGE", example="RENT")
    verification_status: str = Field("Not Verified", description="Income verification status")
    addr_state: str = Field("CA", description="Borrower state code")
    dti: Optional[float] = Field(None, description="Debt-to-income ratio")
    pub_rec: Optional[float] = Field(0, description="Number of derogatory public records")
    pub_rec_bankruptcies: Optional[float] = Field(0, description="Public record bankruptcies")
    tax_liens: Optional[float] = Field(0, description="Number of tax liens")
    acc_now_delinq: Optional[float] = Field(0, description="Currently delinquent accounts")
    delinq_2yrs: Optional[float] = Field(0, description="Delinquencies in last 2 years")
    delinq_amnt: Optional[float] = Field(0, description="Delinquent amount")
    mths_since_last_delinq: Optional[float] = Field(None, description="Months since last delinquency")
    collections_12_mths_ex_med: Optional[float] = Field(0, description="Collections excl. medical in 12 months")
    chargeoff_within_12_mths: Optional[float] = Field(0, description="Charge-offs within 12 months")
    num_accts_ever_120_pd: Optional[float] = Field(None, description="Accounts ever 120+ days past due")
    num_tl_30dpd: Optional[float] = Field(None, description="Accounts 30+ days past due")
    num_tl_90g_dpd_24m: Optional[float] = Field(None, description="Accounts 90+ days past due in 24 months")
    revol_bal: float = Field(0, description="Total revolving balance ($)", example=12500)
    revol_util: Optional[float] = Field(None, description="Revolving utilization (%)")
    bc_util: Optional[float] = Field(None, description="Bank card utilization (%)")
    avg_cur_bal: Optional[float] = Field(None, description="Average current balance")
    tot_cur_bal: Optional[float] = Field(None, description="Total current balance")
    total_bal_ex_mort: Optional[float] = Field(None, description="Total balance excl. mortgage")
    total_bc_limit: Optional[float] = Field(None, description="Total bankcard limit")
    percent_bc_gt_75: Optional[float] = Field(None, description="% bank cards > 75% utilization")
    open_acc: Optional[float] = Field(None, description="Number of open accounts")
    total_acc: Optional[float] = Field(None, description="Total accounts")
    mort_acc: Optional[float] = Field(None, description="Number of mortgage accounts")
    pct_tl_nvr_dlq: Optional[float] = Field(None, description="% trades never delinquent")
    inq_last_6mths: Optional[float] = Field(0, description="Inquiries in last 6 months")
    acc_open_past_24mths: Optional[float] = Field(None, description="Accounts opened in past 24 months")
    loan_status: str = Field("Charged Off", description="Loan status", example="Charged Off")
    credit_history_years: Optional[float] = Field(None, description="Years of credit history (optional, derived from earliest_cr_line if not provided)")

    model_config = {"json_schema_extra": {
        "examples": [{
            "loan_amnt": 15000, "funded_amnt": 15000, "term": "36 months",
            "int_rate": 13.56, "installment": 509.47, "purpose": "debt_consolidation",
            "application_type": "Individual", "grade": "C", "sub_grade": "C3",
            "annual_inc": 75000, "emp_length": "5 years", "home_ownership": "RENT",
            "verification_status": "Verified", "addr_state": "CA", "dti": 22.5,
            "revol_bal": 12500, "loan_status": "Charged Off"
        }]
    }}


class PredictionResponse(BaseModel):
    """Output schema for a single loan prediction."""
    predicted_net_recovery: float = Field(..., description="Predicted net recovery amount ($)")
    recovery_probability: float = Field(..., description="Probability of any recovery (0-1)")
    expected_recovery_if_recovered: float = Field(..., description="Expected amount if recovery occurs ($)")
    risk_tier: str = Field(..., description="Recovery risk tier: High / Medium / Low / Very Low")


class BatchInput(BaseModel):
    """Input schema for batch predictions."""
    loans: list[LoanInput]


class BatchResponse(BaseModel):
    """Output schema for batch predictions."""
    predictions: list[PredictionResponse]
    summary: dict


# ─────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────
def _prepare_features(loan: LoanInput) -> pd.DataFrame:
    """Convert a LoanInput into a feature DataFrame matching training schema."""
    data = loan.model_dump()

    # Derive engineered features
    annual_inc = data.get('annual_inc') or 0
    loan_amnt = data.get('loan_amnt') or 0
    installment = data.get('installment') or 0
    revol_bal = data.get('revol_bal') or 0
    tot_cur_bal = data.get('tot_cur_bal') or 0
    funded_amnt = data.get('funded_amnt') or 0
    dti = data.get('dti') or 0
    int_rate = data.get('int_rate') or 0

    data['loan_to_income'] = loan_amnt / (annual_inc + 1.0)
    data['installment_to_income'] = (installment * 12.0) / (annual_inc + 1.0)
    data['revol_bal_to_income'] = revol_bal / (annual_inc + 1.0)
    data['tot_cur_bal_to_loan'] = (tot_cur_bal or 0) / (loan_amnt + 1.0)
    data['funded_to_loan_ratio'] = funded_amnt / (loan_amnt + 1.0)
    data['debt_burden_index'] = dti * int_rate
    data['revol_util_pct'] = data.get('revol_util') or 0

    # Default credit history if not provided
    if data.get('credit_history_years') is None:
        data['credit_history_years'] = 15.0  # median approximation

    # Clean term
    if data.get('term'):
        data['term'] = str(data['term']).strip()

    # Build DataFrame with exact column order from training
    row = {}
    for col in metadata['feature_cols']:
        if col in data:
            row[col] = data[col]
        else:
            row[col] = np.nan

    df = pd.DataFrame([row])

    # Cast categorical columns to str
    for c in metadata['cat_cols']:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna('Missing')

    return df


def _classify_risk_tier(p_recovery: float, predicted_amount: float) -> str:
    """Assign a risk tier based on recovery probability and predicted amount."""
    if p_recovery >= 0.70 and predicted_amount >= 1000:
        return "High Recovery"
    elif p_recovery >= 0.50 and predicted_amount >= 400:
        return "Medium Recovery"
    elif p_recovery >= 0.25:
        return "Low Recovery"
    else:
        return "Very Low / Write-Off"


def _predict_single(loan: LoanInput) -> PredictionResponse:
    """Run two-stage hurdle prediction for a single loan."""
    df = _prepare_features(loan)

    # Stage 1: Classification
    p_recovery = float(classifier.predict_proba(df)[0, 1])

    # Stage 2: Regression (expected amount if positive)
    e_if_pos = float(max(0, regressor.predict(df)[0]))

    # Hurdle combination
    predicted_net = round(max(0, p_recovery * e_if_pos), 2)

    tier = _classify_risk_tier(p_recovery, predicted_net)

    return PredictionResponse(
        predicted_net_recovery=predicted_net,
        recovery_probability=round(p_recovery, 4),
        expected_recovery_if_recovered=round(e_if_pos, 2),
        risk_tier=tier,
    )


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """API root — info and health."""
    return {
        "service": "CatBoost Net Recovery Prediction API",
        "version": "1.0.0",
        "model": "Two-Stage Hurdle CatBoost (Classifier + Regressor)",
        "target": "net_recovery = recoveries - collection_recovery_fee",
        "status": "healthy",
        "endpoints": {
            "GET /":               "This info page",
            "GET /health":         "Health check",
            "GET /model/metrics":  "Model performance metrics",
            "POST /predict":       "Predict recovery for a single loan",
            "POST /predict/batch": "Predict recovery for multiple loans",
        }
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": classifier is not None and regressor is not None
    }


@app.get("/model/metrics", tags=["Model"])
def model_metrics():
    """Return the model's training/test performance metrics."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")
    return {
        "model_type": "Two-Stage Hurdle CatBoost",
        "metrics": metadata["metrics"],
        "feature_count": len(metadata["feature_cols"]),
        "categorical_features": metadata["cat_cols"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(loan: LoanInput):
    """
    Predict the net recovery amount for a single defaulted loan.

    The model uses borrower and loan attributes to estimate:
    - **recovery_probability**: Likelihood the borrower pays back anything
    - **expected_recovery_if_recovered**: Dollar amount expected if recovery occurs
    - **predicted_net_recovery**: Final prediction = P(recovery) × E(amount|recovery)
    - **risk_tier**: High / Medium / Low / Very Low classification
    """
    try:
        return _predict_single(loan)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """
    Predict net recovery amounts for a batch of defaulted loans.

    Returns individual predictions plus a portfolio-level summary.
    """
    if len(batch.loans) > 1000:
        raise HTTPException(status_code=400, detail="Maximum batch size is 1000 loans")

    try:
        predictions = [_predict_single(loan) for loan in batch.loans]

        amounts = [p.predicted_net_recovery for p in predictions]
        probs = [p.recovery_probability for p in predictions]

        summary = {
            "total_loans": len(predictions),
            "total_predicted_recovery": round(sum(amounts), 2),
            "avg_predicted_recovery": round(np.mean(amounts), 2),
            "avg_recovery_probability": round(np.mean(probs), 4),
            "high_recovery_count": sum(1 for p in predictions if p.risk_tier == "High Recovery"),
            "medium_recovery_count": sum(1 for p in predictions if p.risk_tier == "Medium Recovery"),
            "low_recovery_count": sum(1 for p in predictions if p.risk_tier == "Low Recovery"),
            "writeoff_count": sum(1 for p in predictions if p.risk_tier == "Very Low / Write-Off"),
        }

        return BatchResponse(predictions=predictions, summary=summary)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Batch prediction error: {str(e)}")


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
