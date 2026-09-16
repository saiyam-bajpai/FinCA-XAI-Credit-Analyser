def calculate_emi(principal: float, annual_interest_rate: float = 10.5, tenure_months: int = 36) -> float:
    """
    Calculates monthly Equated Monthly Installment (EMI) using the standard amortization formula:
    E = P * r * (1+r)^n / ((1+r)^n - 1)
    
    :param principal: Loan amount (P)
    :param annual_interest_rate: Annual interest rate in percent (e.g. 10.5)
    :param tenure_months: Loan term in months (n)
    :return: Monthly EMI amount rounded to 2 decimal places
    """
    if principal <= 0 or tenure_months <= 0:
        return 0.0

    monthly_rate = (annual_interest_rate / 12.0) / 100.0

    if monthly_rate == 0:
        return round(principal / tenure_months, 2)

    compounding = (1 + monthly_rate) ** tenure_months
    emi = principal * monthly_rate * compounding / (compounding - 1)
    return round(float(emi), 2)


def calculate_foir_max_loan_limit(
    annual_income: float,
    existing_emis: float,
    annual_interest_rate: float = 10.5,
    tenure_months: int = 36,
    foir_cap_ratio: float = 0.50
) -> dict:
    """
    Calculates Maximum Allowed EMI and Maximum Eligible Loan Amount based on Fixed Obligation to Income Ratio (FOIR).
    Formula: Max_EMI = (Monthly_Income * foir_cap_ratio) - Existing_EMIs
    """
    monthly_income = annual_income / 12.0 if annual_income > 0 else 0.0
    max_allowed_total_emi = monthly_income * foir_cap_ratio
    max_new_emi = max(0.0, max_allowed_total_emi - existing_emis)

    monthly_rate = (annual_interest_rate / 12.0) / 100.0

    if monthly_rate == 0:
        max_loan = max_new_emi * tenure_months
    else:
        compounding = (1 + monthly_rate) ** tenure_months
        max_loan = max_new_emi * (compounding - 1) / (monthly_rate * compounding)

    foir_ratio = (existing_emis / monthly_income) if monthly_income > 0 else 1.0

    return {
        "monthly_income": round(monthly_income, 2),
        "max_allowed_total_emi": round(max_allowed_total_emi, 2),
        "available_monthly_emi_capacity": round(max_new_emi, 2),
        "max_eligible_loan_amount": round(max(0.0, float(max_loan)), 2),
        "foir_ratio": round(foir_ratio, 4)
    }
