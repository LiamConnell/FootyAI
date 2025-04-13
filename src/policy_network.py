import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple
import torch.nn.functional as F



SIGMA_MULTIPLIER = .1


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.tanh(self.fc_mu(x))
        # sigma = SIGMA_MULTIPLIER * torch.exp(self.fc_sigma(x)) + 1e-5
        sigma = SIGMA_MULTIPLIER * F.softplus(self.fc_sigma(x) + .5) + 1e-6
        return mu, sigma