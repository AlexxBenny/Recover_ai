import pandas as pd

def preprocess(df: pd.DataFrame):

    df = df.copy()

    df["has_prior_delinq"] = (
        df["mths_since_last_delinq"]
        .notnull()
        .astype(int)
    )

    df["mths_since_last_delinq"] = (
        df["mths_since_last_delinq"]
        .fillna(999)
    )

    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce"
    )

    df["credit_age_years"] = (
        pd.Timestamp.today() -
        df["earliest_cr_line"]
    ).dt.days / 365.25

    df["income_to_loan_ratio"] = (
        df["annual_inc"]
        / df["loan_amnt"]
    )

    df['delinq_score'] = (
        df['acc_now_delinq']
        + df['delinq_2yrs']
        + df['num_tl_90g_dpd_24m']
        + df['num_accts_ever_120_pd']
    )
    
    df['stress_score'] = (
        df['revol_util']
        + df['bc_util']
    )

    df.drop(
        columns=["earliest_cr_line"],
        inplace=True
    )

    return df