import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import torch

from model import get_model

# --- Loss Functions ---
# For BNN, we need to compute the ELBO loss (NLL + KL Divergence)
kl_loss_fn = bnn.BKLLoss(reduction='mean', last_layer_only=False)
# For both models, the data-driven part of the loss is the cross-entropy
nll_loss_fn = torch.nn.CrossEntropyLoss()

# --- MLflow setup ---
def setup_mlflow(cfg: DictConfig):
    """Sets up the MLflow experiment and run."""
    mlflow.set_tracking_uri("file:///" + hydra.utils.get_original_cwd() + "/mlflow_runs")
    mlflow.set_experiment(cfg.project_name)
    mlflow.start_run(run_name=cfg.run_name)
    
    # Log configuration
    mlflow.log_params(OmegaConf.to_container(cfg.model, resolve=True))
    mlflow.log_params(OmegaConf.to_container(cfg.data, resolve=True))

@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def train(cfg: DictConfig):
    """Main training function."""
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    # Setup MLflow
    setup_mlflow(cfg)

    # --- Dummy Data and Model ---
    print("\n--- Initializing model and data ---")
    model = get_model(cfg.model)
    # Dummy input data (e.g., MNIST-like: [batch, channels, height, width])
    dummy_data = torch.randn(cfg.data.batch_size, 1, 28, 28)
    dummy_labels = torch.randint(0, 10, (cfg.data.batch_size,))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    print("Model and data initialized.")

    # --- Training Loop ---
    print("\n--- Starting training ---")
    for epoch in range(cfg.model.epochs):
        optimizer.zero_grad()
        outputs = model(dummy_data)
        
        # --- Calculate Loss based on model type ---
        nll_loss = nll_loss_fn(outputs, dummy_labels)

        if cfg.model.name == 'bnn':
            kl_loss = kl_loss_fn(model)
            total_loss = nll_loss + cfg.model.kl_weight * kl_loss
            mlflow.log_metrics({"nll_loss": nll_loss.item(), "kl_loss": kl_loss.item()}, step=epoch)
        else: # mc_dropout_dnn
            total_loss = nll_loss

        total_loss.backward()
        optimizer.step()

        # Log metrics to MLflow
        mlflow.log_metric("total_loss", total_loss.item(), step=epoch)
        print(f"Epoch {epoch+1}/{cfg.model.epochs}, Loss: {total_loss.item():.4f}")

    print("--- Training finished ---")

    # Log model to MLflow
    mlflow.pytorch.log_model(model, "model")
    print("Model logged to MLflow.")

    mlflow.end_run()

if __name__ == "__main__":
    train()