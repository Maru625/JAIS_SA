# This is the main entry point for the application.

print("MLOps Project Setup Complete.")
print("\nTo run the training pipeline:")
print("-----------------------------")
print("1. Run a single experiment (using the default cnn model):")
print("   python src/train.py")

print("\n2. Run multiple experiments in parallel (e.g., cnn and transformer):")
print("   python src/train.py --multirun model=cnn,transformer")

print("\n3. To view the results in the MLflow UI:")
print("   mlflow ui --backend-store-uri file:///" + __import__("os").getcwd() + "/mlflow_runs")

print("\n4. To run preprocessing:")
print("   python src/preprocess.py")

print("\n5. To evaluate a trained model (replace with a real run_id):")
print("   python src/evaluate.py --run_id <your_run_id>")