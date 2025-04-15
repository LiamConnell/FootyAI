import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from datetime import datetime
from tqdm import tqdm
from rich import print

from src.torch_soccer_env import TorchSoccerEnv
from src.policy_network import PolicyNetwork
from src.config import N_PLAYERS
from src.visualization import states_to_mp4

VIDEO_PREFIX = f"torch_soccer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Hyperparameters
SIGMA_MULTIPLIER = 0.5
SAMPLE = True  # Whether to sample from the policy or use the mean
BATCH_SIZE = 64  # Number of parallel environments to simulate
NUM_EPISODES = 10000
GAMMA = 0.99
RENDER_EVERY = NUM_EPISODES // 100
LEARNING_RATE = 1e-3

def discount_rewards(rewards_tensor: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute discounted rewards in a fully vectorized way.
    
    Args:
        rewards_tensor: Tensor of shape [T, batch_size] with rewards per time step.
        gamma: Discount factor.
        
    Returns:
        Tensor of shape [T, batch_size] with discounted rewards.
    """
    T = rewards_tensor.size(0)
    # Create discount factors: [1, gamma, gamma^2, ..., gamma^(T-1)]
    discounts = gamma ** torch.arange(T, dtype=rewards_tensor.dtype, device=rewards_tensor.device).view(T, 1)
    
    # Multiply rewards by discount factors and perform a cumulative sum in reversed order
    discounted_temp = rewards_tensor * discounts
    discounted_cumsum = torch.flip(torch.cumsum(torch.flip(discounted_temp, dims=[0]), dim=0), dims=[0])
    
    # Divide by the discount factors to recover the proper discounted sums
    discounted_rewards = discounted_cumsum / discounts
    return discounted_rewards

def update_policy(optimizer: optim.Optimizer,
                  log_probs_a_steps: list,
                  log_probs_b_steps: list,
                  rewards_steps: list,
                  gamma: float,
                  device: str):
    """
    Update the policy network using vectorized discounted rewards.
    """
    # Stack data: shapes [T, batch_size, ...]
    log_probs_a = torch.stack(log_probs_a_steps)  # e.g. [T, batch_size, action_dim]
    log_probs_b = torch.stack(log_probs_b_steps)
    rewards_tensor = torch.stack(rewards_steps)     # [T, batch_size]

    # Compute discounted rewards using the vectorized function
    discounted_rewards = discount_rewards(rewards_tensor, gamma)

    # Optionally normalize rewards per environment
    if discounted_rewards.size(0) > 1:
        rewards_mean = discounted_rewards.mean(dim=0, keepdim=True)
        rewards_std = discounted_rewards.std(dim=0, keepdim=True)
        discounted_rewards = (discounted_rewards - rewards_mean) / (rewards_std + 1e-9)
    
    # Team A receives the rewards and Team B gets the negative
    discounted_rewards_a = discounted_rewards
    discounted_rewards_b = -discounted_rewards

    # Sum log probabilities over action dimensions and compute loss
    loss_a = - (log_probs_a.sum(dim=-1) * discounted_rewards_a).sum(dim=0).mean()
    loss_b = - (log_probs_b.sum(dim=-1) * discounted_rewards_b).sum(dim=0).mean()

    policy_loss = loss_a + loss_b

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

def train_self_play(policy_net: PolicyNetwork,
                    optimizer: optim.Optimizer,
                    batch_size: int,
                    num_episodes: int,
                    render_every: int,
                    device: str,
                    gamma: float = 0.99):
    """
    Train the policy network using self-play in batched environments.
    """
    env = TorchSoccerEnv(batch_size=batch_size, device=device)

    # Pre-create one-hot team identity tensors on the device
    team_a_id = torch.cat([
        torch.ones(batch_size, 1, device=device),
        torch.zeros(batch_size, 1, device=device)
    ], dim=1)
    team_b_id = torch.cat([
        torch.zeros(batch_size, 1, device=device),
        torch.ones(batch_size, 1, device=device)
    ], dim=1)

    for episode in tqdm(range(num_episodes)):
        states = env.reset()

        log_probs_a_steps = []
        log_probs_b_steps = []
        rewards_steps = []

        if episode % render_every == 0:
            rendered_states = [env.to_pydantic(batch_idx=0)]
            
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        while not done.all():
            # Instead of re-creating team ids every time, reuse precreated tensors.
            team_a_inputs = torch.cat([states, team_a_id], dim=1)
            team_b_inputs = torch.cat([states, team_b_id], dim=1)

            mu_a, sigma_a = policy_net(team_a_inputs)
            mu_b, sigma_b = policy_net(team_b_inputs)

            sigma_a = torch.clamp(sigma_a, 0, .5)
            sigma_b = torch.clamp(sigma_b, 0, .5)

            if SAMPLE:
                dist_a = Normal(mu_a, sigma_a * SIGMA_MULTIPLIER)
                actions_a = dist_a.sample()
                batch_log_probs_a = dist_a.log_prob(actions_a)

                dist_b = Normal(mu_b, sigma_b * SIGMA_MULTIPLIER)
                actions_b = dist_b.sample()
                batch_log_probs_b = dist_b.log_prob(actions_b)
            else:
                actions_a = mu_a
                batch_log_probs_a = torch.zeros_like(mu_a, device=device)
                actions_b = mu_b
                batch_log_probs_b = torch.zeros_like(mu_b, device=device)

            # Reshape actions and combine for both teams
            actions_a = actions_a.view(batch_size, N_PLAYERS, 4)
            actions_b = actions_b.view(batch_size, N_PLAYERS, 4)
            actions = torch.cat([actions_a, actions_b], dim=1)

            next_states, batch_rewards, batch_done, _ = env.step(actions)

            log_probs_a_steps.append(batch_log_probs_a)
            log_probs_b_steps.append(batch_log_probs_b)
            rewards_steps.append(batch_rewards)

            states = next_states
            done = done | batch_done

            # Store environment state for rendering (only for first environment)
            if episode % render_every == 0:
                rendered_states.append(env.to_pydantic(batch_idx=0))


        update_policy(optimizer, log_probs_a_steps, log_probs_b_steps, rewards_steps, gamma, device)

        # Render video of gameplay
        if episode % render_every == 0:
            avg_score = torch.mean(env.score, dim=0)
            print(f"Episode {episode+1}/{num_episodes}, Avg Score: {avg_score.cpu().numpy()}, Avg Reward: {np.mean(rewards_steps)}, Avg Abs Reward: {np.mean([np.abs(r) for r in rewards_steps])}")
            os.makedirs(f"videos/{VIDEO_PREFIX}", exist_ok=True)
            states_to_mp4(rendered_states, f"videos/{VIDEO_PREFIX}/episode_{episode+1}.mp4")

    # Save the final model
    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), f"models/{VIDEO_PREFIX}_final.pt")
    print(f"Training complete! Model saved as models/{VIDEO_PREFIX}_final.pt")

def main():
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    os.makedirs("videos", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Setup environment to get observation dimensions.
    env = TorchSoccerEnv(batch_size=1, device=device)
    input_dim = env._get_observation().shape[1] + 2  # Add 2 for team one-hot encoding
    hidden_dim = 128
    output_dim = N_PLAYERS * 4  # x,y movement and x,y kick for each player

    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    train_self_play(
        policy_net=policy_net,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        num_episodes=NUM_EPISODES,
        render_every=RENDER_EVERY,
        device=device,
        gamma=GAMMA
    )

if __name__ == "__main__":
    main()
