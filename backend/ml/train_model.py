import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# Constants & Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
DATASET_PATH = os.path.join(DATA_DIR, "loan_approval_dataset.csv")

FEATURE_COLUMNS = [
    "income",
    "debt",
    "credit_score",
    "loan_amount",
    "loan_term",
    "employment_length",
    "existing_emis",
    "dependents"
]
TARGET_COLUMN = "loan_status"


def generate_synthetic_dataset(num_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic financial dataset for loan approval credit scoring.
    """
    np.random.seed(random_state)

    income = np.random.lognormal(mean=10.8, sigma=0.5, size=num_samples)  # ~$35k - $180k
    income = np.clip(income, 18000, 250000)

    employment_length = np.random.exponential(scale=5, size=num_samples)
    employment_length = np.clip(employment_length, 0, 30).round(1)

    credit_score = np.random.normal(loc=670, scale=80, size=num_samples)
    credit_score = np.clip(credit_score, 300, 850).round().astype(int)

    dependents = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.4, 0.25, 0.2, 0.1, 0.05])

    loan_amount = np.random.uniform(5000, 80000, size=num_samples)
    loan_term = np.random.choice([12, 24, 36, 48, 60], size=num_samples, p=[0.1, 0.2, 0.4, 0.15, 0.15])

    # Debt and Existing Monthly EMIs correlated with income
    debt_ratio = np.random.uniform(0.1, 0.6, size=num_samples)
    debt = (income * debt_ratio).round(2)

    existing_emis = (debt / 24 + np.random.uniform(50, 500, size=num_samples)).round(2)

    # Realistic ground truth logit for loan approval probability
    # Higher credit score, income, employment length -> higher approval chance
    # Higher debt, loan amount, EMIs, dependents -> lower approval chance
    dti_ratio = (existing_emis * 12) / (income + 1e-5)
    
    score_component = (credit_score - 600) / 100.0
    income_component = np.log(income / 30000)
    dti_component = -3.5 * dti_ratio
    loan_to_income = -(loan_amount / (income + 1e-5)) * 0.8
    emp_component = 0.1 * employment_length

    logit = 0.5 + score_component + income_component + dti_component + loan_to_income + emp_component
    prob = 1 / (1 + np.exp(-logit))

    loan_status = (prob > 0.45).astype(int)  # 1: Approved, 0: Rejected

    df = pd.DataFrame({
        "income": income.round(2),
        "debt": debt.round(2),
        "credit_score": credit_score,
        "loan_amount": loan_amount.round(2),
        "loan_term": loan_term,
        "employment_length": employment_length,
        "existing_emis": existing_emis.round(2),
        "dependents": dependents,
        "loan_status": loan_status
    })

    # Add minor synthetic missing values to simulate real preprocessing needs
    missing_indices = np.random.choice(df.index, size=int(0.02 * num_samples), replace=False)
    df.loc[missing_indices, "employment_length"] = np.nan

    return df


def preprocess_data(df: pd.DataFrame, is_training: bool = True, scaler: StandardScaler = None):
    """
    Preprocesses missing values, handles numerical scaling, and returns arrays/scaler.
    """
    df = df.copy()

    # Handle missing values
    if "employment_length" in df.columns:
        median_emp = df["employment_length"].median() if is_training else 3.0
        df["employment_length"] = df["employment_length"].fillna(median_emp)

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None

    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for inference preprocessing.")
        X_scaled = scaler.transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS, index=df.index)

    return X_scaled_df, y, scaler


def train_and_save_model():
    """
    Trains XGBoost model with class imbalance handling, evaluates performance, and serializes artifacts.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Generating/loading dataset...")
    df = generate_synthetic_dataset(num_samples=3000)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset saved to {DATASET_PATH} (Shape: {df.shape})")

    # Split train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET_COLUMN])

    X_train, y_train, scaler = preprocess_data(train_df, is_training=True)
    X_test, y_test, _ = preprocess_data(test_df, is_training=False, scaler=scaler)

    # Class imbalance weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / max(num_pos, 1)

    print(f"Training XGBoost Classifier (Class balance ratio: {scale_pos_weight:.2f})...")
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Save artifacts
    model_path = os.path.join(ARTIFACTS_DIR, "xgboost_model.pkl")
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "scaler_means": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }, meta_path)

    print(f"\nModel saved successfully at: {model_path}")
    print(f"Scaler saved successfully at: {scaler_path}")


if __name__ == "__main__":
    train_and_save_model()
