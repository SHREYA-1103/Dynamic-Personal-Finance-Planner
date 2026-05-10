import numpy as np
import os
import pandas as pd


# GENERATE USER PROFILES
def sample_user_profile():

    # AGE
    age = int(np.random.uniform(22, 60))
    years_worked = max(0, age - 22)

    # USER SEGMENT
    user_segment = np.random.choice(
        ["lower_mid", "mid", "upper_mid"],
        p=[0.4, 0.45, 0.15]
    )

    # INCOME (monthly)
    if user_segment == "lower_mid":
        income = np.random.lognormal(mean=10.4, sigma=0.35)
    elif user_segment == "mid":
        income = np.random.lognormal(mean=11.0, sigma=0.35)
    else:
        income = np.random.lognormal(mean=11.6, sigma=0.4)

    # AGE-BASED GROWTH
    growth_factor = 1 + (years_worked / 40)
    income *= growth_factor

    # RANDOM VARIATION
    income *= np.random.lognormal(0, 0.2)

    # CLIP INCOME
    if np.random.rand() < 0.95:
        income = np.clip(income, 20000, 300000)
    else:
        income = np.clip(income, 300000, 800000)

    income = int(max(income, 20000))

    # INCOME VARIABILITY
    if user_segment == "lower_mid":
        income_var_category = np.random.choice(["medium", "high"], p=[0.6, 0.4])
    elif user_segment == "mid":
        income_var_category = np.random.choice(["low", "medium"], p=[0.5, 0.5])
    else:
        income_var_category = "low"

    sigma_map = {"low": 0.05, "medium": 0.15, "high": 0.30}
    sigma_income = sigma_map[income_var_category]

    # JOB LOSS PROBABILITY
    if income_var_category == "low":
        job_risk_category = "low"
    elif income_var_category == "medium":
        job_risk_category = np.random.choice(["low", "medium"], p=[0.4, 0.6])
    else:
        job_risk_category = np.random.choice(["medium", "high"], p=[0.5, 0.5])

    job_map = {"low": 0.001, "medium": 0.004, "high": 0.008}
    p_job = job_map[job_risk_category]

    # EXPENSES (monthly)
    if income < 50000:
        expense_ratio = np.random.uniform(0.75, 0.95)
    elif income < 120000:
        expense_ratio = np.random.uniform(0.55, 0.75)
    else:
        expense_ratio = np.random.uniform(0.40, 0.65)

    expenses = income * expense_ratio
    expenses *= np.random.lognormal(mean=0, sigma=0.1)
    expenses = int(expenses)

    # SAVINGS RATE
    savings_rate = max(0.05, 1 - (expenses / income))
    savings_rate *= np.random.uniform(0.7, 1.3)
    savings_rate = np.clip(savings_rate, 0.05, 0.8)

    # WEALTH ACCUMULATION
    wealth = 0

    for _ in range(years_worked):

        yearly_income = income * 12 * np.random.uniform(0.85, 1.2)

        yearly_savings = yearly_income * savings_rate * np.random.uniform(0.4, 0.8)

        r = np.random.normal(0.07, 0.10)

        wealth = wealth * (1 + r) + yearly_savings

    # SEGMENT ADJUSTMENT
    if user_segment == "lower_mid":
        wealth *= np.random.uniform(0.4, 0.7)
    elif user_segment == "mid":
        wealth *= np.random.uniform(0.6, 0.9)

    wealth *= np.random.lognormal(mean=0, sigma=0.4)

    if age < 25:
        wealth *= np.random.uniform(0.2, 0.7)

    # SOFT CAP
    threshold = 5e7
    if wealth > threshold:
        wealth = threshold + np.sqrt(wealth - threshold) * 1e4

    wealth = max(0, int(wealth))

    # EMERGENCY BUFFER (months only)
    if income_var_category == "low":
        months = np.random.randint(4, 8)
    elif income_var_category == "medium":
        months = np.random.randint(6, 10)
    else:
        months = np.random.randint(8, 24)

    # RISK APPETITE
    if age < 35 and wealth > 500000:
        risk_profile = np.random.choice(["moderate", "aggressive"], p=[0.3, 0.7])
    elif age > 50:
        risk_profile = np.random.choice(["conservative", "moderate"], p=[0.7, 0.3])
    else:
        risk_profile = np.random.choice(
            ["conservative", "moderate", "aggressive"],
            p=[0.3, 0.5, 0.2]
        )

    risk_map = {
        "conservative": 0.3,
        "moderate": 0.5,
        "aggressive": 0.7
    }

    risk_appetite = risk_map[risk_profile]

    # GOAL HORIZON (months)
    goal_horizon = int((65 - age) * 12 + 12)

    # FINAL PROFILE
    profile = {
        "wealth": wealth,
        "age": age,
        "goal": goal_horizon,
        "income": income,  # monthly
        "income_var": sigma_income,
        "job_loss_prob": p_job,
        "expense": expenses,  # monthly
        "emergency_buffer_months": int(months),
        "risk_appetite": risk_appetite
    }

    return profile


if __name__ == "__main__":
    file_path = "../data/processed/user_profiles.csv"

    if os.path.exists(file_path):
        print("User profiles already exists")

    else:
        profiles = [sample_user_profile() for _ in range(20000)]
        df = pd.DataFrame(profiles)

        df.to_csv("../data/processed/user_profiles.csv", index=False)

        print("User profiles saved at data/processed/user_profiles.csv")