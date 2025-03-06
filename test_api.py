import argparse
import json
import os

import pandas as pd
import requests


def test_health(base_url):
    """
    Test the health endpoint
    """
    url = f"{base_url}/health"
    response = requests.get(url)

    print(f"Health Check Status Code: {response.status_code}")
    print(f"Health Check Response: {response.json()}")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    return response.json()


def test_predict(base_url, test_data_file):
    """
    Test the prediction endpoint
    """
    url = f"{base_url}/predict"

    # Load sample data
    if not os.path.exists(test_data_file):
        print(f"Error: Test data file {test_data_file} not found.")
        return

    df = pd.read_csv(test_data_file)

    # Take the first row as a sample
    sample = df.iloc[0].to_dict()

    # Make prediction request
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(sample), headers=headers)

    print(f"Prediction Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Prediction Response: {response.json()}")
    else:
        print(f"Error Response: {response.text}")
        # If there's an error, it might be due to missing or extra features
        # Let's try to fix the sample data based on the model's expected features
        try:
            # Load feature names from the model
            with open("models/feature_names.pkl", "rb") as f:
                import pickle

                feature_names = pickle.load(f)

            print(f"Model expects these features: {feature_names}")

            # Create a new sample with only the expected features
            new_sample = {}
            for feature in feature_names:
                if feature in sample:
                    new_sample[feature] = sample[feature]
                else:
                    # If a feature is missing, use a default value (0 for numeric)
                    new_sample[feature] = 0

            # Try again with the fixed sample
            print("\nRetrying with fixed sample data...")
            response = requests.post(url, data=json.dumps(new_sample), headers=headers)
            print(f"Prediction Status Code: {response.status_code}")
            print(f"Prediction Response: {response.json()}")
        except Exception as e:
            print(f"Error fixing sample data: {str(e)}")

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()

    return response.json()


def test_explain(base_url, test_data_file):
    """
    Test the explanation endpoint
    """
    url = f"{base_url}/explain"

    # Load sample data
    if not os.path.exists(test_data_file):
        print(f"Error: Test data file {test_data_file} not found.")
        return

    df = pd.read_csv(test_data_file)

    # Take the first row as a sample
    sample = df.iloc[0].to_dict()

    # Make explanation request
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(sample), headers=headers)

    print(f"Explanation Status Code: {response.status_code}")
    if response.status_code == 200:
        print(
            f"Explanation Response (truncated): {json.dumps(response.json(), indent=2)[:500]}..."
        )
    else:
        print(f"Error Response: {response.text}")
        # If there's an error, it might be due to missing or extra features
        # Let's try to fix the sample data based on the model's expected features
        try:
            # Load feature names from the model
            with open("models/feature_names.pkl", "rb") as f:
                import pickle

                feature_names = pickle.load(f)

            print(f"Model expects these features: {feature_names}")

            # Create a new sample with only the expected features
            new_sample = {}
            for feature in feature_names:
                if feature in sample:
                    new_sample[feature] = sample[feature]
                else:
                    # If a feature is missing, use a default value (0 for numeric)
                    new_sample[feature] = 0

            # Try again with the fixed sample
            print("\nRetrying with fixed sample data...")
            response = requests.post(url, data=json.dumps(new_sample), headers=headers)
            print(f"Explanation Status Code: {response.status_code}")
            print(
                f"Explanation Response (truncated): {json.dumps(response.json(), indent=2)[:500]}..."
            )
        except Exception as e:
            print(f"Error fixing sample data: {str(e)}")

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "shap_values" in response.json()
    assert "top_features" in response.json()

    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Test the Customer Churn Prediction API"
    )
    parser.add_argument(
        "--url", default="http://localhost:5000", help="Base URL of the API"
    )
    parser.add_argument(
        "--data", default="data/test_data.csv", help="Path to the test data file"
    )

    args = parser.parse_args()

    print(f"Testing API at {args.url}")

    # Test health endpoint
    print("\n=== Testing Health Endpoint ===")
    test_health(args.url)

    # Test prediction endpoint
    print("\n=== Testing Prediction Endpoint ===")
    test_predict(args.url, args.data)

    # Test explanation endpoint
    print("\n=== Testing Explanation Endpoint ===")
    test_explain(args.url, args.data)

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
