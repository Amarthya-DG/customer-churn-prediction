import argparse
import os
import subprocess
import sys
import time


def check_data_files(train_data_path):
    """
    Check if the data file exists
    """
    if not os.path.exists(train_data_path):
        print(f"Error: Training data file not found at {train_data_path}")
        print(
            "Please place your training dataset in the specified path or provide a custom path."
        )
        return False

    return True


def run_train_model(train_data_path, test_size):
    """
    Run the model training script
    """
    print("\n" + "=" * 50)
    print("STEP 1: TRAINING THE MODEL")
    print("=" * 50)

    cmd = [
        sys.executable,
        "train_model.py",
        "--train_data",
        train_data_path,
        "--test_size",
        str(test_size),
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Print output in real-time
    for line in process.stdout:
        print(line, end="")

    # Wait for the process to complete
    process.wait()

    if process.returncode != 0:
        print("Error in model training. Check the error messages above.")
        for line in process.stderr:
            print(line, end="")
        return False

    return True


def run_explain_model(train_data_path, test_size):
    """
    Run the model explanation script
    """
    print("\n" + "=" * 50)
    print("STEP 2: GENERATING MODEL EXPLANATIONS")
    print("=" * 50)

    cmd = [
        sys.executable,
        "explain_model.py",
        "--train_data",
        train_data_path,
        "--test_size",
        str(test_size),
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Print output in real-time
    for line in process.stdout:
        print(line, end="")

    # Wait for the process to complete
    process.wait()

    if process.returncode != 0:
        print("Error in model explanation. Check the error messages above.")
        for line in process.stderr:
            print(line, end="")
        return False

    return True


def run_api(port=5000):
    """
    Run the API server
    """
    print("\n" + "=" * 50)
    print("STEP 3: STARTING THE API SERVER")
    print("=" * 50)

    cmd = [sys.executable, "app.py", "--port", str(port)]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Wait a bit for the server to start
    time.sleep(2)

    # Check if the process is still running
    if process.poll() is not None:
        print("Error starting the API server. Check the error messages below:")
        for line in process.stderr:
            print(line, end="")
        return False

    print(f"API server running on http://localhost:{port}")
    print("Press Ctrl+C to stop the server")

    try:
        # Print output in real-time
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line, end="")
    except KeyboardInterrupt:
        print("\nStopping the API server...")
        process.terminate()

    return True


def main():
    """
    Main function to run the entire pipeline
    """
    parser = argparse.ArgumentParser(
        description="Run the customer churn prediction pipeline"
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
    parser.add_argument(
        "--skip_train", action="store_true", help="Skip the model training step"
    )
    parser.add_argument(
        "--skip_explain", action="store_true", help="Skip the model explanation step"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the API server on"
    )

    args = parser.parse_args()

    print("Starting Customer Churn Prediction Pipeline")
    print(f"Training data: {args.train_data}")

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Check if data files exist
    if not check_data_files(args.train_data):
        return

    # Run the pipeline steps
    if not args.skip_train:
        if not run_train_model(args.train_data, args.test_size):
            print("Model training failed. Stopping pipeline.")
            return
    else:
        print("\nSkipping model training step...")

    if not args.skip_explain:
        if not run_explain_model(args.train_data, args.test_size):
            print("Model explanation failed. Stopping pipeline.")
            return
    else:
        print("\nSkipping model explanation step...")

    # Run the API server
    run_api(port=args.port)


if __name__ == "__main__":
    main()
