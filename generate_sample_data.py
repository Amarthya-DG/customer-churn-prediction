import os

import numpy as np
import pandas as pd


def generate_sample_data(n_samples=1000, output_file="data/customer_churn.csv"):
    """
    Generate a sample customer churn dataset
    """
    print(f"Generating sample customer churn dataset with {n_samples} samples...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a dictionary to store the data
    data = {
        # Customer demographics
        "CustomerID": np.arange(1, n_samples + 1),
        "Gender": np.random.choice(["Male", "Female"], size=n_samples),
        "SeniorCitizen": np.random.choice([0, 1], size=n_samples),
        "Partner": np.random.choice(["Yes", "No"], size=n_samples),
        "Dependents": np.random.choice(["Yes", "No"], size=n_samples),
        # Services
        "PhoneService": np.random.choice(["Yes", "No"], size=n_samples),
        "MultipleLines": np.random.choice(
            ["Yes", "No", "No phone service"], size=n_samples
        ),
        "InternetService": np.random.choice(
            ["DSL", "Fiber optic", "No"], size=n_samples
        ),
        "OnlineSecurity": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        "OnlineBackup": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        "DeviceProtection": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        "TechSupport": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        "StreamingTV": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        "StreamingMovies": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        # Account information
        "Contract": np.random.choice(
            ["Month-to-month", "One year", "Two year"], size=n_samples
        ),
        "PaperlessBilling": np.random.choice(["Yes", "No"], size=n_samples),
        "PaymentMethod": np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            size=n_samples,
        ),
        # Numeric features
        "Tenure": np.random.randint(0, 72, size=n_samples),  # Months
        "MonthlyCharges": np.random.uniform(18, 118, size=n_samples).round(2),
        "TotalCharges": np.zeros(n_samples),
    }

    # Calculate TotalCharges based on Tenure and MonthlyCharges
    for i in range(n_samples):
        data["TotalCharges"][i] = (data["Tenure"][i] * data["MonthlyCharges"][i]).round(
            2
        )

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Generate the target variable (Churn) based on a simple model
    # Higher churn probability for:
    # - Month-to-month contracts
    # - No online security
    # - Higher monthly charges
    # - Lower tenure

    churn_prob = np.zeros(n_samples)

    # Base churn probability
    churn_prob += 0.1

    # Contract type effect
    churn_prob += np.where(df["Contract"] == "Month-to-month", 0.2, 0)
    churn_prob += np.where(df["Contract"] == "One year", -0.05, 0)
    churn_prob += np.where(df["Contract"] == "Two year", -0.1, 0)

    # Online security effect
    churn_prob += np.where(df["OnlineSecurity"] == "No", 0.1, 0)

    # Tenure effect (normalized to 0-1 range)
    tenure_effect = -0.2 * (df["Tenure"] / 72)
    churn_prob += tenure_effect

    # Monthly charges effect (normalized to 0-1 range)
    charges_effect = 0.15 * ((df["MonthlyCharges"] - 18) / 100)
    churn_prob += charges_effect

    # Internet service effect
    churn_prob += np.where(df["InternetService"] == "Fiber optic", 0.1, 0)

    # Add some randomness
    churn_prob += np.random.normal(0, 0.05, size=n_samples)

    # Clip probabilities to [0, 1] range
    churn_prob = np.clip(churn_prob, 0, 1)

    # Generate binary churn labels
    df["Churn"] = np.random.binomial(1, churn_prob, size=n_samples)
    df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the dataset to CSV
    df.to_csv(output_file, index=False)

    print(f"Sample dataset generated and saved to {output_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Churn distribution: {df['Churn'].value_counts(normalize=True)}")

    return df


if __name__ == "__main__":
    generate_sample_data()
