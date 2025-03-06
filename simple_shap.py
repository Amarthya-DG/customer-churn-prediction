import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Create directories
os.makedirs("explanations", exist_ok=True)


def load_data():
    """
    Load the training and test data
    """
    print("Loading data...")
    train_df = pd.read_csv("data/train_data.csv")
    test_df = pd.read_csv("data/test_data.csv")
    return train_df, test_df


def robust_preprocess(df):
    """
    More robust preprocessing to ensure all data is numeric
    """
    # Make a copy
    processed_df = df.copy()

    # First, handle the target variable
    if "Churn" in processed_df.columns:
        if processed_df["Churn"].dtype == "object":
            processed_df["Churn"] = processed_df["Churn"].map({"Yes": 1, "No": 0})
            # Handle any other values
            processed_df["Churn"] = processed_df["Churn"].fillna(0).astype(float)

    # Identify numeric and categorical columns
    numeric_cols = processed_df.select_dtypes(include=["number"]).columns.tolist()
    if "Churn" in numeric_cols:
        numeric_cols.remove("Churn")

    categorical_cols = processed_df.select_dtypes(include=["object"]).columns.tolist()

    # Handle numeric columns - fill missing values with 0
    for col in numeric_cols:
        processed_df[col] = processed_df[col].fillna(0)

    # Handle categorical columns - convert to one-hot encoding
    for col in categorical_cols:
        # Get dummies and add to dataframe
        dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
        processed_df = pd.concat([processed_df, dummies], axis=1)
        # Drop original column
        processed_df = processed_df.drop(col, axis=1)

    # Ensure all remaining columns are numeric
    for col in processed_df.columns:
        if col != "Churn" and processed_df[col].dtype == "object":
            # Try to convert to numeric, fill non-convertible values with 0
            processed_df[col] = pd.to_numeric(
                processed_df[col], errors="coerce"
            ).fillna(0)

    return processed_df


def main():
    """
    Main function to generate simple SHAP explanations
    """
    print("Starting simple SHAP explanation...")

    # Load data
    try:
        train_df, test_df = load_data()
        print(f"Loaded data: Train shape {train_df.shape}, Test shape {test_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Load model
    try:
        model = load_model("models/churn_model.h5")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        # Robust preprocessing
        print("Preprocessing data...")
        processed_train = robust_preprocess(train_df)
        processed_test = robust_preprocess(test_df)

        # Ensure train and test have the same columns (except Churn)
        train_cols = set(processed_train.columns)
        test_cols = set(processed_test.columns)

        if "Churn" in train_cols:
            train_cols.remove("Churn")
        if "Churn" in test_cols:
            test_cols.remove("Churn")

        # Find common columns
        common_cols = list(train_cols.intersection(test_cols))
        print(f"Using {len(common_cols)} common features")

        # Extract features
        X_train = processed_train[common_cols].values
        X_test = processed_test[common_cols].values

        # Scale the data
        print("Scaling data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Take a small sample for SHAP
        sample_size = min(50, X_train_scaled.shape[0])
        X_train_sample = X_train_scaled[:sample_size]

        test_sample_size = min(5, X_test_scaled.shape[0])
        X_test_sample = X_test_scaled[:test_sample_size]

        print(
            f"Using {sample_size} training samples and {test_sample_size} test samples for SHAP"
        )

        # Create a wrapper function for the model
        def model_predict(x):
            return model.predict(x, verbose=0)

        # Create a simple explainer
        print("Creating SHAP explainer...")
        background = shap.kmeans(X_train_sample, 10)  # Use kmeans for background
        explainer = shap.KernelExplainer(model_predict, background)

        # Calculate SHAP values for a few test samples
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_test_sample)

        # Create summary plot
        print("Creating summary plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_test_sample, feature_names=common_cols, show=False
        )
        plt.tight_layout()
        plt.savefig("explanations/simple_summary_plot.png")
        plt.close()

        # Create bar plot
        print("Creating bar plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_test_sample,
            feature_names=common_cols,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plt.savefig("explanations/simple_bar_plot.png")
        plt.close()

        # Create a simple feature importance plot
        print("Creating feature importance plot...")
        # Calculate mean absolute SHAP values for each feature
        feature_importance = np.abs(shap_values).mean(0)

        # Ensure feature_importance is a 1D array
        if hasattr(feature_importance, "shape") and len(feature_importance.shape) > 1:
            feature_importance = feature_importance.mean(axis=1)

        # Create a bar chart of feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(common_cols)), feature_importance, align="center")
        plt.yticks(range(len(common_cols)), common_cols)
        plt.xlabel("Mean |SHAP Value|")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("explanations/feature_importance.png")
        plt.close()

        print("\nSHAP explanation completed successfully!")
        print("Explanations saved to the 'explanations' directory:")
        print("- explanations/simple_summary_plot.png")
        print("- explanations/simple_bar_plot.png")
        print("- explanations/feature_importance.png")

    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
