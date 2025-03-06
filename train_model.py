import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential

np.random.seed(42)
tf.random.set_seed(42)


def load_data(train_file_path):
    """
    Load the customer churn train dataset
    """
    print(f"Loading training data from {train_file_path}")
    train_df = pd.read_csv(train_file_path)
    print(f"Training dataset shape: {train_df.shape}")

    return train_df


def explore_data(df, name="dataset"):
    """
    Perform basic exploratory data analysis
    """
    print(f"\n{name.capitalize()} Overview:")
    print(df.info())

    print(f"\n{name.capitalize()} Summary Statistics:")
    print(df.describe())

    print(f"\n{name.capitalize()} Churn Distribution:")
    print(df["Churn"].value_counts(normalize=True))

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.countplot(x="Churn", data=df)
    plt.title(f"{name.capitalize()} Churn Distribution")
    plt.savefig(f"plots/{name}_churn_distribution.png")

    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=["number"])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{name.capitalize()} Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"plots/{name}_correlation_heatmap.png")

    return


def preprocess_data(train_df, test_df=None):
    """
    Preprocess the data for model training
    """
    print("\nPreprocessing data...")

    processed_train_df = train_df.copy()
    if test_df is not None:
        processed_test_df = test_df.copy()

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

    os.makedirs("models", exist_ok=True)
    with open("models/numeric_columns.pkl", "wb") as f:
        pickle.dump(numeric_cols, f)

    with open("models/categorical_columns.pkl", "wb") as f:
        pickle.dump(categorical_cols, f)

    for col in numeric_cols:
        Q1 = processed_train_df[col].quantile(0.25)
        Q3 = processed_train_df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for outliers (using 3*IQR to be conservative)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # Cap extreme values in train set
        processed_train_df[col] = processed_train_df[col].clip(lower_bound, upper_bound)

        # Cap extreme values in test set
        if test_df is not None:
            processed_test_df[col] = processed_test_df[col].clip(
                lower_bound, upper_bound
            )

        print(
            f"Capped extreme values in {col} between {lower_bound:.2f} and {upper_bound:.2f}"
        )

    # Store column means and modes for imputation
    column_means = {}
    column_modes = {}

    # Handle missing values if any in train set
    if processed_train_df.isnull().sum().sum() > 0:
        print(
            f"Missing values found in train set: {processed_train_df.isnull().sum().sum()}"
        )

        # Fill numeric columns with mean
        for col in numeric_cols:
            if processed_train_df[col].isnull().any():
                col_mean = processed_train_df[col].mean()
                column_means[col] = col_mean
                processed_train_df[col] = processed_train_df[col].fillna(col_mean)

        # Fill categorical columns with mode
        for col in categorical_cols:
            if processed_train_df[col].isnull().any():
                col_mode = processed_train_df[col].mode()[0]
                column_modes[col] = col_mode
                processed_train_df[col] = processed_train_df[col].fillna(col_mode)
    else:
        # Even if no missing values, calculate means and modes for future use
        for col in numeric_cols:
            column_means[col] = processed_train_df[col].mean()

        for col in categorical_cols:
            column_modes[col] = processed_train_df[col].mode()[0]

    # Save column means and modes
    with open("models/column_means.pkl", "wb") as f:
        pickle.dump(column_means, f)

    with open("models/column_modes.pkl", "wb") as f:
        pickle.dump(column_modes, f)

    if test_df is not None:
        # Handle missing values if any in test set
        if processed_test_df.isnull().sum().sum() > 0:
            print(
                f"Missing values found in test set: {processed_test_df.isnull().sum().sum()}"
            )

            # Fill numeric columns with train means
            for col in numeric_cols:
                if (
                    col in processed_test_df.columns
                    and processed_test_df[col].isnull().any()
                ):
                    processed_test_df[col] = processed_test_df[col].fillna(
                        column_means[col]
                    )

            # Fill categorical columns with train modes
            for col in categorical_cols:
                if (
                    col in processed_test_df.columns
                    and processed_test_df[col].isnull().any()
                ):
                    processed_test_df[col] = processed_test_df[col].fillna(
                        column_modes[col]
                    )

    # Convert categorical variables to dummy variables
    if categorical_cols:
        print(f"Converting categorical columns: {categorical_cols}")
        processed_train_df = pd.get_dummies(
            processed_train_df, columns=categorical_cols, drop_first=True
        )
        if test_df is not None:
            processed_test_df = pd.get_dummies(
                processed_test_df, columns=categorical_cols, drop_first=True
            )

        # Save dummy columns for later use
        dummy_columns = [col for col in processed_train_df.columns if col != "Churn"]
        with open("models/dummy_columns.pkl", "wb") as f:
            pickle.dump(dummy_columns, f)

    if test_df is not None:
        # Ensure train and test have the same columns (except target)
        train_features = processed_train_df.drop("Churn", axis=1).columns
        test_features = processed_test_df.drop("Churn", axis=1).columns

        # Find columns in train but not in test
        train_only = set(train_features) - set(test_features)
        if train_only:
            print(f"Columns in train but not in test: {train_only}")
            # Add these columns to test with 0 values
            for col in train_only:
                processed_test_df[col] = 0

        # Find columns in test but not in train
        test_only = set(test_features) - set(train_features)
        if test_only:
            print(f"Columns in test but not in train: {test_only}")
            # Add these columns to train with 0 values
            for col in test_only:
                processed_train_df[col] = 0

    # Convert target variable to binary if it's not already
    if processed_train_df["Churn"].dtype == "object":
        print("Converting Churn to binary in train set")
        processed_train_df["Churn"] = processed_train_df["Churn"].map(
            {"Yes": 1, "No": 0}
        )

    if test_df is not None:
        if processed_test_df["Churn"].dtype == "object":
            print("Converting Churn to binary in test set")
            processed_test_df["Churn"] = processed_test_df["Churn"].map(
                {"Yes": 1, "No": 0}
            )

    # Check for any remaining NaN values
    if processed_train_df.isnull().sum().sum() > 0:
        print(
            f"Warning: There are still {processed_train_df.isnull().sum().sum()} missing values in the train set"
        )
        # Fill any remaining NaNs with 0
        processed_train_df = processed_train_df.fillna(0)

    if test_df is not None:
        if processed_test_df.isnull().sum().sum() > 0:
            print(
                f"Warning: There are still {processed_test_df.isnull().sum().sum()} missing values in the test set"
            )
            # Fill any remaining NaNs with 0
            processed_test_df = processed_test_df.fillna(0)

    # Split features and target
    X_train = processed_train_df.drop("Churn", axis=1)
    y_train = processed_train_df["Churn"]

    if test_df is not None:
        X_test = processed_test_df.drop("Churn", axis=1)
        y_test = processed_test_df["Churn"]

    # Check for infinite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    if test_df is not None:
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale the features using RobustScaler which is less sensitive to outliers
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if test_df is not None:
        X_test_scaled = scaler.transform(X_test)

    # Save the scaler for later use
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save feature names for later use
    feature_names = X_train.columns.tolist()
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    if test_df is not None:
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    else:
        return X_train_scaled, y_train, feature_names


def create_model(input_dim):
    """
    Create a deep learning model for churn prediction
    """
    print("\nCreating model...")

    model = Sequential()

    # Input layer
    model.add(Dense(64, input_dim=input_dim, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Hidden layers
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    # Use a smaller learning rate and add clipnorm to prevent exploding gradients
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    # Compile model with a small epsilon to prevent numerical instability
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1),
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    print(model.summary())

    return model


def plot_training_history(history):
    """
    Plot the training history
    """
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/training_history.png")
    print("Training history plot saved to plots/training_history.png")


def train_model(train_data_path, test_size=0.2):
    """
    Train a deep learning model with SMOTE and early stopping for class imbalance.
    """
    # Load the training data
    train_data = pd.read_csv(train_data_path)

    # Preprocess the data
    X_train_scaled, y_train, feature_names = preprocess_data(train_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_scaled, y_train, test_size=test_size, random_state=42
    )

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Create the deep learning model
    model = create_model(X_train_smote.shape[1])

    # Implement early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored metric.
        verbose=1,
    )

    # Train the model with early stopping
    history = model.fit(
        X_train_smote,
        y_train_smote,
        epochs=5,  # Increased number of epochs
        batch_size=32,  # Adjust as needed
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],  # Add early stopping callback
        verbose=1,
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy: {accuracy:.4f}")

    # Save the trained model
    model_filename = "customer_churn_model.h5"
    model.save(model_filename)
    print(f"Trained model saved to {model_filename}")

    return model_filename


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set
    """
    print("\nEvaluating model on test set...")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Churned", "Churned"],
        yticklabels=["Not Churned", "Churned"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    print("Confusion matrix saved to plots/confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("plots/roc_curve.png")
    print("ROC curve saved to plots/roc_curve.png")

    return y_pred_proba


def save_model(model):
    """
    Save the trained model
    """
    print("\nSaving model...")

    # Save the model in TensorFlow SavedModel format
    model.save("models/churn_model")

    # Save the model in HDF5 format
    model.save("models/churn_model.h5")

    print("Model saved successfully.")


def main():
    """
    Main function to train the model
    """
    parser = argparse.ArgumentParser(
        description="Train a customer churn prediction model"
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

    print("Starting Model Training...")
    print(f"Training data: {args.train_data}")

    # Train the model
    train_model(args.train_data, args.test_size)


if __name__ == "__main__":
    main()
