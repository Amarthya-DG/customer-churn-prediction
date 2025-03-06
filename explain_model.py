import argparse
import pickle

import numpy as np
import pandas as pd
import shap
from tensorflow.keras.models import load_model


def load_data(train_file_path):
    """
    Load the customer churn train dataset
    """
    print(f"Loading training data from {train_file_path}")
    train_df = pd.read_csv(train_file_path)
    print(f"Training dataset shape: {train_df.shape}")

    return train_df


def preprocess_data_for_shap(train_df):
    """
    Preprocess the data for SHAP analysis
    """
    print("\nPreprocessing data for SHAP analysis...")

    processed_train_df = train_df.copy()

    numeric_cols = processed_train_df.select_dtypes(include=["number"]).columns.tolist()
    if "Churn" in numeric_cols:
        numeric_cols.remove("Churn")

    categorical_cols = processed_train_df.select_dtypes(
        include=["object"]
    ).columns.tolist()
    if "Churn" in categorical_cols:
        categorical_cols.remove("Churn")

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    if processed_train_df.isnull().sum().sum() > 0:
        print(
            f"Missing values found in train set: {processed_train_df.isnull().sum().sum()}"
        )
        for col in numeric_cols:
            if processed_train_df[col].isnull().any():
                col_mean = processed_train_df[col].mean()
                processed_train_df[col] = processed_train_df[col].fillna(col_mean)

        for col in categorical_cols:
            if processed_train_df[col].isnull().any():
                col_mode = processed_train_df[col].mode()[0]
                processed_train_df[col] = processed_train_df[col].fillna(col_mode)

    if categorical_cols:
        print(f"Converting categorical columns: {categorical_cols}")
        processed_train_df = pd.get_dummies(
            processed_train_df, columns=categorical_cols, drop_first=True
        )

    if processed_train_df["Churn"].dtype == "object":
        print("Converting Churn to binary in train set")
        processed_train_df["Churn"] = processed_train_df["Churn"].map(
            {"Yes": 1, "No": 0}
        )

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    try:
        processed_train_df = processed_train_df[feature_names + ["Churn"]]
    except KeyError as e:
        print(f"Error: Missing feature in datasets: {e}")
        print("Using available features instead...")
        common_features = list(
            set(processed_train_df.columns) & set(feature_names + ["Churn"])
        )

        processed_train_df = processed_train_df[common_features]

        feature_names = [f for f in common_features if f != "Churn"]

    X = processed_train_df.drop("Churn", axis=1)
    y = processed_train_df["Churn"]

    X_scaled = scaler.transform(X)

    return X_scaled, y, feature_names


def load_trained_model():
    """
    Load the trained model
    """
    print("\nLoading trained model...")
    model = load_model("customer_churn_model.h5")
    return model


def explain_with_shap(model, X, feature_names):
    """
    Explain the model predictions using SHAP
    """
    print("\nExplaining model predictions with SHAP...")

    sample_size = min(100, X.shape[0])
    X_sample = X[:sample_size]

    def model_predict(x):
        return model.predict(x)

    explainer = shap.KernelExplainer(model_predict, X_sample)

    shap_values = explainer.shap_values(X_sample)

    return shap_values, explainer


def explain_model(train_data_path, test_size=0.2):
    """
    Explain the deep learning model using SHAP values.
    """
    train_data = pd.read_csv(train_data_path)

    X_scaled, y, feature_names = preprocess_data_for_shap(train_data)

    model = load_trained_model()

    shap_values, explainer = explain_with_shap(model, X_scaled, feature_names)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = list(zip(feature_names, mean_abs_shap))

    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance


def main():
    """
    Main function to explain the model
    """
    parser = argparse.ArgumentParser(
        description="Explain a customer churn prediction model using SHAP"
    )

    parser.add_argument(
        "--train_data",
        type=str,
        default="data/train_data.csv",
        help="Path to the training data CSV file",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Size of the test set (as a fraction of the training data)",
    )

    args = parser.parse_args()

    print("Starting Model Explanation...")
    print(f"Training data: {args.train_data}")

    feature_importance = explain_model(args.train_data, args.test_size)

    print("\nFeature Importances:")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
