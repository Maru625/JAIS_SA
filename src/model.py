import torch
import torch.nn as nn
import torchbnn as bnn

def get_model(model_config):
    """Factory function to create a model based on the config."""
    if model_config.name == "bnn":
        return BNN(model_config)
    elif model_config.name == "mc_dropout_dnn":
        return McDropoutDnn(model_config)
    else:
        raise ValueError(f"Unknown model: {model_config.name}")

class McDropoutDnn(nn.Module):
    """A standard DNN with Dropout layers for Monte Carlo Dropout."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_units),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_units, config.hidden_units),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_units, config.output_size),
        )

    def forward(self, x):
        # Flatten the input if it's image-like
        x = x.view(x.size(0), -1)
        return self.network(x)

class BNN(nn.Module):
    """A Bayesian Neural Network using the torchbnn library."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.network = bnn.BnnSequential(
            bnn.BayesLinear(prior_mu=config.prior_mu, prior_sigma=config.prior_sigma, in_features=config.input_size, out_features=config.hidden_units),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=config.prior_mu, prior_sigma=config.prior_sigma, in_features=config.hidden_units, out_features=config.output_size)
        )

    def forward(self, x):
        # Flatten the input if it's image-like
        x = x.view(x.size(0), -1)
        return self.network(x)