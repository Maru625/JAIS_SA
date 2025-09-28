import mlflow
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

def evaluate_mc_dropout(model, test_data, inference_samples):
    """Performs evaluation for an MC Dropout model."""
    print(f"\n--- Evaluating MC Dropout model with {inference_samples} samples ---")
    # Set model to train mode to enable dropout during inference
    model.train()
    
    predictions = []
    with torch.no_grad():
        for _ in range(inference_samples):
            outputs = model(test_data)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            predictions.append(softmax_outputs.numpy())
    
    # Shape: (inference_samples, num_test_samples, num_classes)
    predictions = np.array(predictions)
    
    # Calculate mean prediction and uncertainty
    mean_predictions = predictions.mean(axis=0)
    predictive_uncertainty = predictions.var(axis=0).mean(axis=1) # Mean variance across classes
    
    final_predictions = mean_predictions.argmax(axis=1)
    
    return final_predictions, predictive_uncertainty

def evaluate_bnn(model, test_data):
    """Performs evaluation for a BNN model."""
    print("\n--- Evaluating BNN model ---")
    # For BNN, we can do a single forward pass for a point estimate
    # Or multiple passes to sample from the posterior
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs, 1)
    # Uncertainty can be estimated by sampling, but we'll keep it simple here
    # and return zero as a placeholder.
    uncertainty = np.zeros(len(predicted))
    return predicted.numpy(), uncertainty

@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def evaluate(cfg: DictConfig):
    """Main evaluation function."""
    # It's assumed you will modify the config to point to the run you want to evaluate
    # For now, we'll just use the default model config
    run_id = "<your_run_id>" # IMPORTANT: Replace with a real run_id from MLflow
    if run_id == "<your_run_id>":
        print("Please open `src/evaluate.py` and replace `<your_run_id>` with a real run ID from your MLflow experiments.")
        return

    print(f"Evaluating model from run_id: {run_id}")

    # Load model from the specified run
    logged_model = f"runs:/{run_id}/model"
    try:
        model = mlflow.pytorch.load_model(logged_model)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create dummy test data
    dummy_test_data = torch.randn(100, 1, 28, 28)
    dummy_test_labels = torch.randint(0, 10, (100,))

    # --- Evaluation based on model type ---
    # We can infer the model type from the loaded model's config
    if hasattr(model, 'config') and model.config.name == 'mc_dropout_dnn':
        final_predictions, uncertainty = evaluate_mc_dropout(model, dummy_test_data, model.config.inference_samples)
    elif hasattr(model, 'config') and model.config.name == 'bnn':
        final_predictions, uncertainty = evaluate_bnn(model, dummy_test_data)
    else:
        print("Warning: Could not determine model type from config. Performing standard evaluation.")
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_test_data)
            _, final_predictions = torch.max(outputs, 1)
            final_predictions = final_predictions.numpy()
            uncertainty = np.zeros(len(final_predictions))

    accuracy = (final_predictions == dummy_test_labels.numpy()).mean()
    avg_uncertainty = uncertainty.mean()

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Average Predictive Uncertainty: {avg_uncertainty:.4f}")

    # Log evaluation metrics to the original run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "avg_test_uncertainty": avg_uncertainty
        })
    print("Evaluation metrics logged to MLflow.")

if __name__ == "__main__":
    evaluate()