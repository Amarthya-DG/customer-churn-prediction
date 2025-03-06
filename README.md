# Customer Churn Prediction with Deep Learning

This project implements a deep learning model for customer churn prediction, with model explainability using SHAP (SHapley Additive exPlanations) and deployment using Docker.

## Project Structure

```
.
├── app.py                  # Flask API for model deployment
├── data/                   # Directory for data files
│   ├── train_data.csv      # Training dataset
├── Dockerfile              # Dockerfile for containerization
├── docker-compose.yml      # Docker Compose configuration
├── explain_model.py        # Script for model explainability using SHAP
├── models/                 # Directory for saved models and artifacts
├── plots/                  # Directory for plots and visualizations
│   └── shap/               # SHAP visualizations
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── run_pipeline.py         # Script to run the entire pipeline
├── test_api.py             # Script to test the API
└── train_model.py          # Script for data preprocessing and model training
```

## Requirements

- Python 3.9+
- TensorFlow 2.12.0
- SHAP 0.41.0
- Flask 2.3.2
- Docker (for containerized deployment)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd customer-churn-prediction
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Place your training and test datasets in the `data/` directory:
   ```
   mkdir -p data
   # Copy your datasets to data/train_data.csv and data/test_data.csv
   ```

## Dataset

The customer churn datasets should be in CSV format with the following structure:
- A column named 'Churn' (target variable) with values 'Yes'/'No' or 1/0
- Various customer features (demographic, usage, etc.)

You can provide two separate datasets:
1. `data/train_data.csv` - Used for training the model i used one one dataset and splited into training and testing
2. `data/test_data.csv` - Used for testing the model

## Quick Start

The easiest way to run the entire pipeline is to use the `run_pipeline.py` script:

```
python run_pipeline.py
```

This will:
1. Check if the datasets exist
2. Train the deep learning model
3. Generate SHAP explanations
4. Start the Flask API locally

Additional options:
```
python run_pipeline.py --skip-training     # Skip model training
python run_pipeline.py --skip-explanation  # Skip SHAP explanation
python run_pipeline.py --skip-api          # Skip API deployment
python run_pipeline.py --use-docker        # Deploy API using Docker
python run_pipeline.py --train-data path/to/train.csv --test-data path/to/test.csv  # Specify custom dataset paths
```

## Detailed Usage

### 1. Training the Model

To train the deep learning model:

```
python train_model.py
```

This will:
- Load and preprocess the training and test data
- Train a deep learning model
- Evaluate the model performance on the test set
- Save the model and artifacts to the `models/` directory
- Generate performance plots in the `plots/` directory

### 2. Explaining the Model

To generate SHAP explanations for the model:

```
python explain_model.py
```

This will:
- Load the trained model
- Generate SHAP values and visualizations using the test data
- Save SHAP plots to the `plots/shap/` directory

### 3. Running the API Locally

To run the Flask API locally:

```
python app.py
```

The API will be available at http://localhost:5000 with the following endpoints:
- `/` - API documentation
- `/health` - Health check endpoint
- `/predict` - Prediction endpoint (POST)
- `/explain` - Explanation endpoint with SHAP values (POST)

### 4. Testing the API

To test the API:

```
python test_api.py --url http://localhost:5000 --data data/test_data.csv
```

### 5. Docker Deployment

To deploy the application using Docker:

```
docker-compose up -d
```

This will:
- Build the Docker image
- Start the container
- Expose the API on port 5000

## API Endpoints

### Prediction Endpoint

**URL**: `/predict`
**Method**: POST
**Request Body**: JSON object with customer features
**Response**: Prediction result with probability

Example request:
```json
{
  "feature1": value1,
  "feature2": value2,
  ...
}
```

Example response:
```json
{
  "prediction": 1,
  "probability": 0.85,
  "message": "Customer is likely to churn"
}
```

### Explanation Endpoint

**URL**: `/explain`
**Method**: POST
**Request Body**: JSON object with customer features
**Response**: Prediction result with SHAP values

Example response:
```json
{
  "prediction": 1,
  "probability": 0.85,
  "shap_values": {
    "feature1": 0.2,
    "feature2": -0.1,
    ...
  },
  "top_features": [
    {"feature": "feature1", "importance": 0.2},
    {"feature": "feature3", "importance": 0.15},
    ...
  ]
}
```

## Model Architecture

The deep learning model consists of:
- Input layer with the number of features
- 3 hidden layers with 64, 32, and 16 neurons respectively
- Batch normalization and dropout for regularization
- Output layer with sigmoid activation for binary classification

## Model Explainability

SHAP (SHapley Additive exPlanations) is used to explain the model predictions. SHAP values represent the contribution of each feature to the prediction, providing both global and local interpretability.

