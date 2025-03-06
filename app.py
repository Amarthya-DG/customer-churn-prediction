import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
scaler = None
feature_names = None
explainer = None


def load_artifacts():
    """
    Load all artifacts needed for prediction
    """
    global model, scaler, feature_names, explainer

    logger.info("Loading model artifacts...")

    # Load model
    model_path = "customer_churn_model.h5"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = load_model(model_path)
    logger.info("Model loaded successfully")

    # Load scaler
    scaler_path = os.path.join("models", "scaler.pkl")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found at {scaler_path}")
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully")

    # Load feature names
    feature_names_path = os.path.join("models", "feature_names.pkl")
    if not os.path.exists(feature_names_path):
        logger.error(f"Feature names file not found at {feature_names_path}")
        raise FileNotFoundError(f"Feature names file not found at {feature_names_path}")

    with open(feature_names_path, "rb") as f:
        feature_names = pickle.load(f)
    logger.info("Feature names loaded successfully")

    # Try to load a small sample of training data for SHAP background
    try:
        train_file_path = os.path.join("data", "train_data.csv")
        if os.path.exists(train_file_path):
            logger.info("Loading training data for SHAP background")
            train_df = pd.read_csv(train_file_path)

            # Process the data to match the model's expected input
            # Handle categorical variables
            categorical_cols = train_df.select_dtypes(
                include=["object"]
            ).columns.tolist()
            if "Churn" in categorical_cols:
                categorical_cols.remove("Churn")

            if categorical_cols:
                train_df = pd.get_dummies(
                    train_df, columns=categorical_cols, drop_first=True
                )

            # Convert target variable to binary if it's not already
            if "Churn" in train_df.columns and train_df["Churn"].dtype == "object":
                train_df["Churn"] = train_df["Churn"].map({"Yes": 1, "No": 0})

            # Extract features
            if "Churn" in train_df.columns:
                X_train = train_df.drop("Churn", axis=1)
            else:
                X_train = train_df

            # Ensure we only use the features that were used in training
            common_features = list(
                set(X_train.columns).intersection(set(feature_names))
            )
            if len(common_features) < len(feature_names):
                logger.warning(
                    f"Only {len(common_features)} out of {len(feature_names)} features found in training data"
                )

            X_train = X_train[common_features]

            # Fill any missing columns with zeros
            for feature in feature_names:
                if feature not in X_train.columns:
                    X_train[feature] = 0

            # Reorder columns to match feature_names
            X_train = X_train[feature_names]

            # Scale the features
            X_train_scaled = scaler.transform(X_train)

            # Create a small background dataset for SHAP
            sample_size = min(100, X_train.shape[0])
            background_data = X_train_scaled[:sample_size]
        else:
            # If no training data is available, use zeros as background
            logger.warning("Training data not found. Using zeros as SHAP background.")
            background_data = np.zeros((10, len(feature_names)))
    except Exception as e:
        logger.error(f"Error loading training data for SHAP: {str(e)}")
        # Use zeros as background if there's an error
        background_data = np.zeros((10, len(feature_names)))

    # Initialize SHAP explainer
    def model_predict(x):
        return model.predict(x)

    explainer = shap.KernelExplainer(model_predict, background_data)
    logger.info("SHAP explainer initialized successfully")

    logger.info("All artifacts loaded successfully")


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint
    """
    try:
        # Get input data from request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input data to DataFrame
        input_df = pd.DataFrame(data, index=[0])

        # Check if all required features are present
        missing_features = set(feature_names) - set(input_df.columns)
        if missing_features:
            return (
                jsonify({"error": f"Missing features: {list(missing_features)}"}),
                400,
            )

        # Reorder columns to match feature_names
        input_df = input_df[feature_names]

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0][0]

        # Convert prediction to binary and probability
        churn_probability = float(prediction)
        churn_prediction = 1 if churn_probability >= 0.5 else 0

        # Prepare response
        response = {
            "prediction": int(churn_prediction),
            "probability": churn_probability,
            "message": "Customer is likely to churn"
            if churn_prediction == 1
            else "Customer is likely to stay",
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """
    Explanation endpoint using SHAP
    """
    try:
        # Get input data from request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input data to DataFrame
        input_df = pd.DataFrame(data, index=[0])

        # Check if all required features are present
        missing_features = set(feature_names) - set(input_df.columns)
        if missing_features:
            return (
                jsonify({"error": f"Missing features: {list(missing_features)}"}),
                400,
            )

        # Reorder columns to match feature_names
        input_df = input_df[feature_names]

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0][0]

        # Get SHAP values
        shap_values = explainer.shap_values(input_scaled)

        # Calculate mean SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create feature importance dictionary
        feature_importance = {}
        for i, feature in enumerate(feature_names):
            feature_importance[feature] = mean_abs_shap[i]

        # Sort features by absolute SHAP value
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Prepare response
        response = {
            "prediction": int(1 if prediction >= 0.5 else 0),
            "probability": float(prediction),
            "shap_values": feature_importance,
            "top_features": [
                {"feature": f[0], "importance": f[1]} for f in sorted_features[:5]
            ],
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during explanation: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    """
    Root endpoint with API documentation
    """
    api_docs = {
        "name": "Customer Churn Prediction API",
        "version": "1.0.0",
        "description": "API for predicting customer churn using deep learning",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API documentation"},
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint",
            },
            {
                "path": "/predict",
                "method": "POST",
                "description": "Make churn predictions",
                "request_body": "JSON object with customer features",
                "response": "Prediction result with probability",
            },
            {
                "path": "/explain",
                "method": "POST",
                "description": "Explain churn predictions using SHAP",
                "request_body": "JSON object with customer features",
                "response": "Prediction result with SHAP values",
            },
        ],
    }

    return jsonify(api_docs)


def preprocess_input(input_data):
    """
    Preprocess the input data for prediction
    """
    # Convert input to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = pd.DataFrame(input_data)

    # Load feature names
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    # Load the scaler
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load the categorical columns
    try:
        with open("models/categorical_columns.pkl", "rb") as f:
            categorical_columns = pickle.load(f)
        print(f"Loaded categorical columns: {categorical_columns}")
    except FileNotFoundError:
        print("No categorical columns file found. Inferring from data types.")
        categorical_columns = input_df.select_dtypes(
            include=["object"]
        ).columns.tolist()

    # Load the numeric columns
    try:
        with open("models/numeric_columns.pkl", "rb") as f:
            numeric_columns = pickle.load(f)
        print(f"Loaded numeric columns: {numeric_columns}")
    except FileNotFoundError:
        print("No numeric columns file found. Inferring from data types.")
        numeric_columns = input_df.select_dtypes(include=["number"]).columns.tolist()

    # Handle missing values
    if input_df.isnull().sum().sum() > 0:
        print(f"Missing values found in input: {input_df.isnull().sum().sum()}")

        # Load means and modes for imputation
        try:
            with open("models/column_means.pkl", "rb") as f:
                column_means = pickle.load(f)

            with open("models/column_modes.pkl", "rb") as f:
                column_modes = pickle.load(f)

            # Fill numeric columns with means
            for col in numeric_columns:
                if col in input_df.columns and input_df[col].isnull().any():
                    if col in column_means:
                        input_df[col] = input_df[col].fillna(column_means[col])
                    else:
                        print(f"Warning: No mean value for {col}. Using 0 instead.")
                        input_df[col] = input_df[col].fillna(0)

            # Fill categorical columns with modes
            for col in categorical_columns:
                if col in input_df.columns and input_df[col].isnull().any():
                    if col in column_modes:
                        input_df[col] = input_df[col].fillna(column_modes[col])
                    else:
                        print(
                            f"Warning: No mode value for {col}. Using 'unknown' instead."
                        )
                        input_df[col] = input_df[col].fillna("unknown")

        except FileNotFoundError:
            print("Warning: Missing means/modes files. Using basic imputation.")
            # Basic imputation
            for col in numeric_columns:
                if col in input_df.columns and input_df[col].isnull().any():
                    input_df[col] = input_df[col].fillna(0)

            for col in categorical_columns:
                if col in input_df.columns and input_df[col].isnull().any():
                    input_df[col] = input_df[col].fillna("unknown")

    # One-hot encode categorical variables
    if categorical_columns:
        # Load expected dummy columns if available
        try:
            with open("models/dummy_columns.pkl", "rb") as f:
                expected_dummies = pickle.load(f)
            print(f"Loaded {len(expected_dummies)} expected dummy columns")

            # One-hot encode
            input_df = pd.get_dummies(
                input_df, columns=categorical_columns, drop_first=True
            )

            # Add missing dummy columns
            for col in expected_dummies:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Remove extra dummy columns
            extra_cols = [
                col
                for col in input_df.columns
                if col not in expected_dummies and col not in numeric_columns
            ]
            if extra_cols:
                print(f"Removing extra columns: {extra_cols}")
                input_df = input_df.drop(columns=extra_cols)

        except FileNotFoundError:
            print("Warning: No dummy columns file found. Using basic one-hot encoding.")
            input_df = pd.get_dummies(
                input_df, columns=categorical_columns, drop_first=True
            )

    # Ensure all required features are present
    missing_features = [f for f in feature_names if f not in input_df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for feature in missing_features:
            input_df[feature] = 0

    # Select only the features used by the model
    input_df = input_df[feature_names]

    # Scale the features
    scaled_input = scaler.transform(input_df)

    return scaled_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the churn prediction API server")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the API server on",
    )

    args = parser.parse_args()

    # Load artifacts before starting the server
    load_artifacts()

    # Start the Flask app
    app.run(host="0.0.0.0", port=args.port, debug=False)
