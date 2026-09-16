import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# Base paths relative to backend/ml/train_model.py
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def preprocess_and_feature_engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()

    # Normalize column names: strip spaces and convert to lowercase
    data.columns = [str(c).strip().lower() for c in data.columns]

    def find_col(candidates):
        for candidate in candidates:
            if candidate in data.columns:
                return candidate
        return None

    # Dynamic column mapping across standard financial datasets
    inc_col = find_col(['annual_inc', 'annualinc', 'income_annum', 'applicantincome', 'income', 'annual_income'])
    loan_col = find_col(['loan_amnt', 'loanamnt', 'loan_amount', 'loanamount'])
    fico_low_col = find_col(['fico_range_low', 'ficorangelow', 'cibil_score', 'credit_score', 'credit_history'])
    fico_high_col = find_col(['fico_range_high', 'ficorangehigh'])
    dti_col = find_col(['dti', 'foir', 'foir_ratio', 'debt_to_income'])
    emi_col = find_col(['installment', 'emi', 'calculated_emi'])
    status_col = find_col(['loan_status', 'target', 'status', 'approved'])

    print("\n--- DETECTED COLUMNS ---")
    print(f"Income Column       : {inc_col}")
    print(f"Loan Amount Column  : {loan_col}")
    print(f"Credit Score Column : {fico_low_col}")
    print(f"Target Status Column: {status_col}\n")

    if not inc_col or not status_col:
        print("❌ Available columns in CSV:", list(data.columns))
        raise KeyError("Could not automatically map required Income or Target status columns.")

    # 1. Annual Income
    if 'coapplicantincome' in data.columns and inc_col == 'applicantincome':
        data['income'] = (
            pd.to_numeric(data['applicantincome'], errors='coerce').fillna(0) +
            pd.to_numeric(data['coapplicantincome'], errors='coerce').fillna(0)
        ) * 12
    elif inc_col == 'applicantincome':
        data['income'] = pd.to_numeric(data[inc_col], errors='coerce').fillna(0) * 12
    else:
        data['income'] = pd.to_numeric(data[inc_col], errors='coerce')

    data['income'] = data['income'].fillna(data['income'].median())

    # 2. Loan Amount
    if loan_col:
        data['loan_amount'] = pd.to_numeric(data[loan_col], errors='coerce')
        if data['loan_amount'].median() < 1000:  # Handle datasets in thousands
            data['loan_amount'] = data['loan_amount'] * 1000
    else:
        data['loan_amount'] = 10000.0
    data['loan_amount'] = data['loan_amount'].fillna(data['loan_amount'].median())

    # 3. Credit Score (CIBIL / FICO)
    if fico_low_col and fico_high_col:
        f_low = pd.to_numeric(data[fico_low_col], errors='coerce')
        f_high = pd.to_numeric(data[fico_high_col], errors='coerce')
        data['credit_score'] = ((f_low + f_high) / 2.0).fillna(680.0)
    elif fico_low_col:
        data['credit_score'] = pd.to_numeric(data[fico_low_col], errors='coerce')
        if data['credit_score'].nunique() <= 3:  # Binary credit history indicator
            data['credit_score'] = data['credit_score'].apply(lambda x: 750.0 if x == 1 else 550.0)
    else:
        data['credit_score'] = 680.0
    data['credit_score'] = data['credit_score'].fillna(680.0)

    # 4. Debt-to-Income / FOIR
    if dti_col:
        dti_vals = pd.to_numeric(data[dti_col], errors='coerce')
        if dti_vals.median() > 1.0:
            dti_vals = dti_vals / 100.0
        data['foir_ratio'] = dti_vals.clip(0.0, 1.0).fillna(0.35)
    else:
        data['foir_ratio'] = 0.35

    # 5. Calculated EMI
    if emi_col:
        data['calculated_emi'] = pd.to_numeric(data[emi_col], errors='coerce').fillna(data['loan_amount'] / 36)
    else:
        data['calculated_emi'] = data['loan_amount'] / 36

    # 6. Binary Target (1 = Approved/Good, 0 = Default/Rejected)
    good_terms = {'fully paid', 'current', 'approved', '1', 'y', 'yes'}
    data['target'] = data[status_col].astype(str).str.strip().str.lower().apply(
        lambda x: 1 if x in good_terms else 0
    )

    # Derived Features
    monthly_income = np.maximum(data['income'] / 12.0, 1.0)
    data['emi_to_income_ratio'] = (data['calculated_emi'] / monthly_income).clip(0.0, 2.0)
    data['max_eligible_loan'] = np.maximum(0, (monthly_income * 0.50) * 36)

    feature_cols = [
        'income',
        'loan_amount',
        'credit_score',
        'foir_ratio',
        'calculated_emi',
        'max_eligible_loan',
        'emi_to_income_ratio',
    ]

    return data[feature_cols], data['target']


def calculate_ks_stat(y_true, y_probs):
    df = pd.DataFrame({'y_true': y_true, 'prob': y_probs}).sort_values(by='prob', ascending=False)
    df['stat'] = df['y_true'].cumsum() / df['y_true'].sum()
    df['non_stat'] = (1 - df['y_true']).cumsum() / (1 - df['y_true']).sum()
    return np.max(np.abs(df['stat'] - df['non_stat']))


def find_dataset_file() -> Path:
    """Searches for common CSV names in project data folders."""
    possible_paths = [
        PROJECT_ROOT / "data" / "loan_approval_dataset.csv",
        PROJECT_ROOT / "data" / "lendingclub_data.csv",
        PROJECT_ROOT.parent / "data" / "loan_approval_dataset.csv",
        PROJECT_ROOT.parent / "data" / "lendingclub_data.csv",
        BASE_DIR / "data" / "loan_approval_dataset.csv",
        BASE_DIR / "data" / "lendingclub_data.csv",
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def main():
    dataset_path = find_dataset_file()

    if not dataset_path:
        print("\n⚠️ No dataset CSV found! Place your CSV inside the `data/` folder.")
        return

    print(f"[1/5] Loading and preprocessing dataset from: {dataset_path}")
    df_raw = pd.read_csv(dataset_path, low_memory=False)
    X, y = preprocess_and_feature_engineer(df_raw)

    print("[2/5] Performing 80/20 Train-Test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

    print("[3/5] Training XGBoost Classifier...")
    scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    base_xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    base_xgb.fit(X_train, y_train, verbose=False)

    print("[4/5] Calibrating probabilities (5-fold cross-validation)...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_xgb,
        method="sigmoid",
        cv=5
    )
    calibrated_model.fit(X_train, y_train)

    # Performance Metrics
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_probs)
    ks = calculate_ks_stat(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)

    print("\n================ TEST SET METRICS ================")
    print(f"ROC-AUC Score    : {auc:.4f}")
    print(f"Gini Coefficient : {2 * auc - 1:.4f}")
    print(f"KS Statistic     : {ks:.4f}")
    print(f"Brier Score Loss : {brier:.4f}")

    print("\n[5/5] Building TreeSHAP explainer & saving artifacts...")
    shap_explainer = shap.TreeExplainer(base_xgb)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, ARTIFACTS_DIR / "finca_calibrated_model.joblib")
    joblib.dump(base_xgb, ARTIFACTS_DIR / "finca_base_xgb.joblib")
    joblib.dump(shap_explainer, ARTIFACTS_DIR / "finca_shap_explainer.joblib")
    joblib.dump(X.columns.tolist(), ARTIFACTS_DIR / "feature_names.joblib")

    print(f"\n✅ Success! Saved production artifacts to: {ARTIFACTS_DIR}\n")


if __name__ == "__main__":
    main()