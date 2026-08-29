from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from preprocessing import preprocess
import joblib

app = FastAPI(
    title="Recovery Prediction Service"
)

classifier = joblib.load("models/loan_recovery_probability_model_v1.joblib")

class PredictionRequest(BaseModel):

    loan_amnt: float
    funded_amnt: float
    term: str
    int_rate: float
    installment: float
    purpose: str
    application_type: str

    grade: str
    sub_grade: str

    annual_inc: float
    emp_length: str
    home_ownership: str
    verification_status: str
    addr_state: str
    dti: float

    earliest_cr_line: str

    pub_rec: int
    pub_rec_bankruptcies: float
    tax_liens: float

    acc_now_delinq: int
    delinq_2yrs: int
    delinq_amnt: float
    mths_since_last_delinq: float | None = None

    collections_12_mths_ex_med: float
    chargeoff_within_12_mths: float

    num_accts_ever_120_pd: float
    num_tl_30dpd: float
    num_tl_90g_dpd_24m: float

    revol_bal: float
    revol_util: float
    bc_util: float

    avg_cur_bal: float
    tot_cur_bal: float

    total_bal_ex_mort: float
    total_bc_limit: float

    percent_bc_gt_75: float

    open_acc: int
    total_acc: int
    mort_acc: float

    pct_tl_nvr_dlq: float

    inq_last_6mths: int
    acc_open_past_24mths: float

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(request : PredictionRequest):
    df = pd.DataFrame([request.model_dump()])

    processed = preprocess(df)

    probability = (
        classifier
        .predict_proba(processed)[0][1]
    )

    # expected_amount = np.expm1(
    #     regressor.predict(processed)[0]
    # )

    # expected_recovery_value = (
    #     probability
    #     * expected_amount
    # )

    return {
        "recovery_probability":
            float(probability)

        # "expected_recovery_amount":
        #     float(expected_amount),

        # "expected_recovery_value":
        #     float(expected_recovery_value)
    }