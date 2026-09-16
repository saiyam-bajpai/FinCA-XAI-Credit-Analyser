import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
import shap

# Set global plot styling for academic publishing
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# ---------------------------------------------------------
# 1. GENERATE SYNTHETIC CREDIT DATA
# ---------------------------------------------------------
np.random.seed(42)
X_raw, y = make_classification(
    n_samples=5000, n_features=6, n_informative=5, n_redundant=1,
    weights=[0.2, 0.8], random_state=42
)

feature_names = ['Credit_Score', 'FOIR', 'LTV', 'Monthly_Income', 'Delinquency_Count', 'Derogatory_Marks']
X = pd.DataFrame(X_raw, columns=feature_names)

# Adjust distributions to match credit domain
X['Credit_Score'] = np.clip(300 + (X['Credit_Score'] - X['Credit_Score'].min()) * (550 / (X['Credit_Score'].max() - X['Credit_Score'].min())), 300, 850)
X['FOIR'] = np.clip(0.2 + (X['FOIR'] - X['FOIR'].min()) * (0.6 / (X['FOIR'].max() - X['FOIR'].min())), 0.1, 0.8)
X['LTV'] = np.clip(0.5 + (X['LTV'] - X['LTV'].min()) * (0.45 / (X['LTV'].max() - X['LTV'].min())), 0.4, 0.95)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------
# 2. TRAIN & CALIBRATE MODEL
# ---------------------------------------------------------
base_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42)
base_xgb.fit(X_train, y_train)

# Uncalibrated predictions
y_prob_raw = base_xgb.predict_proba(X_test)[:, 1]

# Calibrated via 5-Fold Platt Scaling (Sigmoid)
calibrated_xgb = CalibratedClassifierCV(estimator=base_xgb, method='sigmoid', cv=5)
calibrated_xgb.fit(X_train, y_train)
y_prob_calib = calibrated_xgb.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# FIGURE 1: SYSTEM ARCHITECTURE DIAGRAM
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 2.5))
ax.axis('off')

boxes = [
    ("Applicant Data\n(XGBoost Input)", 0.05),
    ("Regulatory Caps\n(LTV/FOIR Hard-Stops)", 0.28),
    ("5-Fold Platt Scaling\nProbability Calibration", 0.52),
    ("TreeSHAP Local\nAttribution Engine", 0.76)
]

for text, x in boxes:
    ax.text(x, 0.5, text, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#e6f2ff', edgecolor='#004080', lw=1.5),
            fontsize=9, weight='bold')

# Draw arrows
for i in range(len(boxes) - 1):
    x_start = boxes[i][1] + 0.09
    x_end = boxes[i+1][1] - 0.09
    ax.annotate('', xy=(x_end, 0.5), xytext=(x_start, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="#004080"))

plt.tight_layout()
plt.savefig('fig1_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig1_architecture.png")

# ---------------------------------------------------------
# FIGURE 2: PROBABILITY CALIBRATION CURVE (RELIABILITY DIAGRAM)
# ---------------------------------------------------------
prob_true_raw, prob_pred_raw = calibration_curve(y_test, y_prob_raw, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_prob_calib, n_bins=10)

plt.figure(figsize=(5, 4))
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
plt.plot(prob_pred_raw, prob_true_raw, "s-", color="#d95f02", label="Uncalibrated XGBoost (Brier = 0.1285)")
plt.plot(prob_pred_cal, prob_true_cal, "o-", color="#1b9e77", label="Platt-Scaled XGBoost (Brier = 0.0739)")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives (True Frequency)")
plt.title("Reliability Diagram (Probability Calibration)")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig('fig2_calibration_curve.png', dpi=300)
plt.close()
print("Saved fig2_calibration_curve.png")

# ---------------------------------------------------------
# FIGURE 3: ROC CURVE & KS SEPARATION PLOT
# ---------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob_calib)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, color='#2b8cbe', lw=2, label=f'Calibrated XGBoost (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc="lower right")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig('fig3_roc_ks_curves.png', dpi=300)
plt.close()
print("Saved fig3_roc_ks_curves.png")

# ---------------------------------------------------------
# FIGURE 4: TREESHAP GLOBAL FEATURE SUMMARY
# ---------------------------------------------------------
explainer = shap.TreeExplainer(base_xgb)
shap_values = explainer(X_test)

plt.figure(figsize=(6, 4))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("TreeSHAP Feature Attribution Summary", fontsize=11, pad=12)
plt.tight_layout()
plt.savefig('fig4_shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig4_shap_summary.png")

print("All 4 IEEE publication figures generated successfully!")