# import torch
# import torch.nn as nn
# from torch.distributions import Normal
# from typing import Tuple
# import torch.nn.functional as F

# SIGMA_MULTIPLIER = 1


# class PolicyNetwork(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc_mu = nn.Linear(hidden_dim, output_dim)
#         self.fc_sigma = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         mu = self.tanh(self.fc_mu(x))
#         # sigma = SIGMA_MULTIPLIER * torch.exp(self.fc_sigma(x)) + 1e-5
#         sigma = SIGMA_MULTIPLIER * F.softplus(self.fc_sigma(x) + .5) + 1e-6
#         return mu, sigma

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple
import torch.nn.functional as F

# Consider increasing SIGMA_MULTIPLIER if needed, or making sigma range wider
SIGMA_MULTIPLIER = 1.0
LOG_SIG_MIN = -20  # Lower bound for log standard deviation
LOG_SIG_MAX = 2   # Upper bound for log standard deviation


class EnhancedPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_hidden_layers: int = 4):
        super(EnhancedPolicyNetwork, self).__init__()

        # Dynamically create hidden layers
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU()) # Or nn.Tanh(), nn.GELU(), etc.

        # Hidden layers
        for _ in range(num_hidden_layers - 1): # -1 because the input layer counts as one
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            # Consider adding Layer Normalization here for stability in deeper nets
            # layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU()) # Or another activation

        # Create the sequential block for shared layers
        self.shared_layers = nn.Sequential(*layers)

        # Output heads for mean and log standard deviation
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_log_sigma = nn.Linear(hidden_dim, output_dim) # Output log_sigma is often more stable

        # Action scaling (if your actions are not in [-1, 1]) - Tanh outputs [-1, 1]
        # If your environment expects actions in [a, b], you'll need to scale mu later.
        # Example: action = action_low + (mu + 1.0) * 0.5 * (action_high - action_low)

        # Optional: Initialize weights differently (e.g., orthogonal)
        self._initialize_weights()

    def _initialize_weights(self):
        # Example: Initialize layers with Xavier Uniform and small biases
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.01)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.xavier_uniform_(self.fc_log_sigma.weight)
        nn.init.constant_(self.fc_log_sigma.bias, -1.0) # Initialize log_sigma towards smaller values initially


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pass through shared layers
        shared_features = self.shared_layers(x)

        # Calculate mean
        mu = self.fc_mu(shared_features)
        mu = torch.tanh(mu) # Constrain mean to [-1, 1]

        # Calculate log standard deviation
        log_sigma = self.fc_log_sigma(shared_features)

        # Clamp log_sigma for stability (prevents very large/small variances)
        log_sigma = torch.clamp(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)

        # Calculate standard deviation
        sigma = torch.exp(log_sigma) * SIGMA_MULTIPLIER

        # Add a small epsilon for numerical stability if needed, although clamping helps
        # sigma = sigma + 1e-6

        return mu, sigma
    

PolicyNetwork = EnhancedPolicyNetwork


class PlayerPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_hidden_layers: int = 4):
        super(PlayerPolicyNetwork, self).__init__()

        # Dynamically create hidden layers
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU()) # Or nn.Tanh(), nn.GELU(), etc.

        # Hidden layers
        for _ in range(num_hidden_layers - 1): # -1 because the input layer counts as one
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            # Consider adding Layer Normalization here for stability in deeper nets
            # layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU()) # Or another activation

        # Create the sequential block for shared layers
        self.shared_layers = nn.Sequential(*layers)

        # Output heads for mean and log standard deviation
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_log_sigma = nn.Linear(hidden_dim, output_dim) # Output log_sigma is often more stable

        # Action scaling (if your actions are not in [-1, 1]) - Tanh outputs [-1, 1]
        # If your environment expects actions in [a, b], you'll need to scale mu later.
        # Example: action = action_low + (mu + 1.0) * 0.5 * (action_high - action_low)

        # Optional: Initialize weights differently (e.g., orthogonal)
        self._initialize_weights()

    def _initialize_weights(self):
        # Example: Initialize layers with Xavier Uniform and small biases
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.01)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.xavier_uniform_(self.fc_log_sigma.weight)
        nn.init.constant_(self.fc_log_sigma.bias, -1.0) # Initialize log_sigma towards smaller values initially


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pass through shared layers
        shared_features = self.shared_layers(x)

        # Calculate mean
        mu = self.fc_mu(shared_features)
        mu = torch.tanh(mu) # Constrain mean to [-1, 1]

        # Calculate log standard deviation
        log_sigma = self.fc_log_sigma(shared_features)

        # Clamp log_sigma for stability (prevents very large/small variances)
        log_sigma = torch.clamp(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)

        # Calculate standard deviation
        sigma = torch.exp(log_sigma) * SIGMA_MULTIPLIER

        # Add a small epsilon for numerical stability if needed, although clamping helps
        # sigma = sigma + 1e-6

        return mu, sigma