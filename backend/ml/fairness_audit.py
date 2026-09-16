import numpy as np
import pandas as pd
import joblib
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, true_positive_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load your existing model and test dataset
model = joblib.load("backend/ml/xgboost_model.pkl")
X_test = pd.read_csv("backend/ml/X_test.csv")
y_test = pd.read_csv("backend/ml/y_test.csv")

# 2. Define your sensitive attribute (e.g., Age < 30 vs Age >= 30)
# Assuming 'age' is a column in your dataset
sensitive_feature = X_test['age'].apply(lambda x: 'Young (<30)' if x < 30 else 'Older (>=30)')

# Get model predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 3. Calculate Fairness Metrics across both groups
metrics = {
    'accuracy': accuracy_score,
    'selection_rate': selection_rate, # Approval rate
    'false_positive_rate': false_positive_rate,
    'true_positive_rate': true_positive_rate
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_feature
)

print("=== FAIRNESS AUDIT REPORT ===")
print(metric_frame.by_group)

# 4. Calculate Disparate Impact Ratio (80% Rule)
approval_rates = metric_frame.by_group['selection_rate']
disparate_impact_ratio = approval_rates['Young (<30)'] / approval_rates['Older (>=30)']

print(f"\nDisparate Impact Ratio: {disparate_impact_ratio:.4f}")
if disparate_impact_ratio < 0.8:
    print("⚠️ WARNING: Algorithmic bias detected against younger applicants (< 0.80 rule)!")
else:
    print("✅ Model passes the 80% Rule for Demographic Parity.")

# 5. Fix Bias using ThresholdOptimizer
postprocessor = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds", # Enforce equal error rates
    predict_method="predict_proba"
)

# Fit the fair optimizer
postprocessor.fit(X_test, y_test, sensitive_features=sensitive_feature)

# Predict with fair thresholds
fair_y_pred = postprocessor.predict(X_test, sensitive_features=sensitive_feature)

# Re-audit the fair model
fair_metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=fair_y_pred,
    sensitive_features=sensitive_feature
)

print("\n=== POST-MITIGATION FAIRNESS REPORT ===")
print(fair_metric_frame.by_group)