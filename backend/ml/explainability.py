import os
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "artifacts")
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "xgboost_model.pkl")
DEFAULT_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

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


class XAIEngine:
    """
    Explainable AI (XAI) Engine for FinCA Credit Scoring using SHAP and Counterfactual Optimization.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, scaler_path: str = DEFAULT_SCALER_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.explainer = None
        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Model or Scaler artifact not found at '{self.model_path}' or '{self.scaler_path}'. "
                "Please run `python train_model.py` first."
            )

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

    def _preprocess_single_record(self, record: Dict[str, Any]) -> pd.DataFrame:
        """
        Ensures all feature columns exist, orders them, and scales numerical values.
        """
        df = pd.DataFrame([record])
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        df = df[FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(df)
        return pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)

    def get_global_feature_importance(self) -> Dict[str, float]:
        """
        Extracts global feature importance scores from the trained XGBoost model.
        """
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        
        importance_dict = {}
        for idx in sorted_indices:
            feature_name = FEATURE_COLUMNS[idx]
            importance_dict[feature_name] = round(float(importances[idx]), 4)

        return importance_dict

    def explain_instance(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates full explainable credit decision output for a single applicant record.
        """
        raw_df = pd.DataFrame([applicant_data])[FEATURE_COLUMNS]
        scaled_df = self._preprocess_single_record(applicant_data)

        # Predict probability
        prob_approved = float(self.model.predict_proba(scaled_df)[0][1])
        status = "Approved" if prob_approved >= 0.50 else "Rejected"
        credit_rating_score = int(300 + (prob_approved * 550))  # Scale 300-850 credit rating

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(scaled_df)
        if isinstance(shap_values, list):
            # Binary classifier SHAP values list [class_0, class_1]
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 2:
            sv = shap_values[0]
        else:
            sv = shap_values[0]

        base_value = float(self.explainer.expected_value[1]) if isinstance(self.explainer.expected_value, (list, np.ndarray)) else float(self.explainer.expected_value)

        # Format feature attributions
        attributions = []
        positive_factors = []
        negative_factors = []

        for idx, col in enumerate(FEATURE_COLUMNS):
            val = float(raw_df[col].iloc[0])
            impact = float(sv[idx])
            direction = "positive" if impact > 0 else "negative" if impact < 0 else "neutral"

            item = {
                "feature": col,
                "raw_value": val,
                "shap_impact": round(impact, 4),
                "direction": direction
            }
            attributions.append(item)

            if impact > 0:
                positive_factors.append(item)
            elif impact < 0:
                negative_factors.append(item)

        # Sort factors by magnitude
        positive_factors = sorted(positive_factors, key=lambda x: x["shap_impact"], reverse=True)
        negative_factors = sorted(negative_factors, key=lambda x: x["shap_impact"])

        # Counterfactual Logic if status is Rejected or score is low
        counterfactuals = self.generate_counterfactuals(applicant_data, current_prob=prob_approved)

        return {
            "applicant_summary": applicant_data,
            "decision": {
                "status": status,
                "approval_probability": round(prob_approved, 4),
                "finCA_credit_score": credit_rating_score,
                "base_expectation": round(base_value, 4)
            },
            "global_feature_importance": self.get_global_feature_importance(),
            "local_attributions": attributions,
            "top_positive_factors": positive_factors[:3],
            "top_negative_factors": negative_factors[:3],
            "counterfactual_recommendations": counterfactuals
        }

    def generate_counterfactuals(self, applicant_data: Dict[str, Any], current_prob: float, target_prob: float = 0.55) -> List[Dict[str, Any]]:
        """
        Calculates exact actionable profile modifications to flip 'Rejected' into 'Approved'.
        """
        if current_prob >= target_prob:
            return [{
                "feature": "overall_profile",
                "recommended_action": "Your profile already satisfies loan approval criteria.",
                "target_value": None,
                "estimated_time_to_fix": "Immediate",
                "impact_on_approval": "Sufficient"
            }]

        recommendations = []
        temp_data = applicant_data.copy()

        # Actionable feature 1: Reduce existing EMIs
        current_emis = float(temp_data.get("existing_emis", 0))
        if current_emis > 100:
            for step in range(50, int(current_emis), 50):
                test_rec = temp_data.copy()
                test_rec["existing_emis"] = max(0, current_emis - step)
                scaled_test = self._preprocess_single_record(test_rec)
                new_prob = float(self.model.predict_proba(scaled_test)[0][1])

                if new_prob >= target_prob or step >= current_emis * 0.5:
                    time_to_fix = f"{max(1, int(step / 150))} - {max(2, int(step / 100))} Months"
                    recommendations.append({
                        "feature": "existing_emis",
                        "current_value": f"${current_emis:,.2f}",
                        "target_value": f"${test_rec['existing_emis']:,.2f}",
                        "recommended_action": f"Reduce monthly EMIs by ${step:,.2f}",
                        "estimated_time_to_fix": time_to_fix,
                        "projected_approval_probability": round(new_prob, 4)
                    })
                    break

        # Actionable feature 2: Reduce Loan Amount (e.g. increase down payment)
        current_loan = float(temp_data.get("loan_amount", 0))
        if current_loan > 5000:
            for pct in [0.05, 0.10, 0.15, 0.20, 0.25]:
                test_rec = temp_data.copy()
                reduced_loan = current_loan * (1 - pct)
                test_rec["loan_amount"] = reduced_loan
                scaled_test = self._preprocess_single_record(test_rec)
                new_prob = float(self.model.predict_proba(scaled_test)[0][1])

                if new_prob >= target_prob or pct >= 0.25:
                    recommendations.append({
                        "feature": "loan_amount",
                        "current_value": f"${current_loan:,.2f}",
                        "target_value": f"${reduced_loan:,.2f}",
                        "recommended_action": f"Reduce requested loan amount by {int(pct*100)}% (or increase down payment by ${current_loan * pct:,.2f})",
                        "estimated_time_to_fix": "Instant / Immediate",
                        "projected_approval_probability": round(new_prob, 4)
                    })
                    break

        # Actionable feature 3: Pay down total debt
        current_debt = float(temp_data.get("debt", 0))
        if current_debt > 1000:
            debt_reduction = min(current_debt * 0.25, 10000)
            test_rec = temp_data.copy()
            test_rec["debt"] = current_debt - debt_reduction
            scaled_test = self._preprocess_single_record(test_rec)
            new_prob = float(self.model.predict_proba(scaled_test)[0][1])

            time_to_fix = f"{max(3, int(debt_reduction / 1500))} Months"
            recommendations.append({
                "feature": "debt",
                "current_value": f"${current_debt:,.2f}",
                "target_value": f"${test_rec['debt']:,.2f}",
                "recommended_action": f"Pay down overall debt by ${debt_reduction:,.2f}",
                "estimated_time_to_fix": time_to_fix,
                "projected_approval_probability": round(new_prob, 4)
            })

        # Actionable feature 4: Increase Credit Score
        current_cs = int(temp_data.get("credit_score", 600))
        if current_cs < 750:
            target_cs = min(850, current_cs + 40)
            test_rec = temp_data.copy()
            test_rec["credit_score"] = target_cs
            scaled_test = self._preprocess_single_record(test_rec)
            new_prob = float(self.model.predict_proba(scaled_test)[0][1])

            recommendations.append({
                "feature": "credit_score",
                "current_value": str(current_cs),
                "target_value": str(target_cs),
                "recommended_action": f"Improve credit score by {target_cs - current_cs} points through on-time payments",
                "estimated_time_to_fix": "3 - 6 Months",
                "projected_approval_probability": round(new_prob, 4)
            })

        # Return top 3 recommendations sorted by highest projected approval probability
        recommendations = sorted(recommendations, key=lambda x: x["projected_approval_probability"], reverse=True)
        return recommendations[:3]


if __name__ == "__main__":
    import json
    
    # Quick test run
    print("Testing XAIEngine...")
    try:
        xai = XAIEngine()
        sample_applicant = {
            "income": 45000.0,
            "debt": 22000.0,
            "credit_score": 590,
            "loan_amount": 35000.0,
            "loan_term": 36,
            "employment_length": 1.5,
            "existing_emis": 750.0,
            "dependents": 2
        }

        explanation = xai.explain_instance(sample_applicant)
        print("\n--- XAI Explanation Output ---")
        print(f"Decision: {explanation['decision']['status']} (Probability: {explanation['decision']['approval_probability']})")
        print(f"FinCA Credit Score: {explanation['decision']['finCA_credit_score']}")
        print("\nTop Negative Factors (Drivers of Rejection):")
        for factor in explanation['top_negative_factors']:
            print(f" - {factor['feature']} (SHAP impact: {factor['shap_impact']})")

        print("\nTop 3 Counterfactual Recommendations:")
        for idx, rec in enumerate(explanation['counterfactual_recommendations'], 1):
            print(f" {idx}. {rec['recommended_action']} | Estimated Time to Fix: {rec['estimated_time_to_fix']}")

    except Exception as e:
        print(f"Error executing XAIEngine: {e}")
