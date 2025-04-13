import torch
import torch.nn as nn
import torch.nn.functional as F

class SoccerPolicyNetwork(nn.Module):
    def __init__(self, input_size=44, hidden_size=128, output_size=20):
        super(SoccerPolicyNetwork, self).__init__()
        
        # Network architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Value head for PPO
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Policy head
        policy = self.fc3(x)
        
        # Value head
        value = self.value_head(x)
        
        return policy, value
    
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            policy, _ = self.forward(state)
            
            if deterministic:
                action = policy
            else:
                # Add noise for exploration
                action = policy + torch.randn_like(policy) * 0.1
                
            # Clip actions to valid range
            action = torch.clamp(action, -2, 2)
            
        return action 