import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from datetime import datetime
from tqdm import tqdm
from rich import print
from google.cloud import storage
from pathlib import Path

from .torch_soccer_env import TorchSoccerEnv
from src.policy_network import PolicyNetwork
from src.config import N_PLAYERS, GAME_DURATION
from src.visualization import states_to_mp4

VIDEO_PREFIX = f"v2_torch_soccer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# GCS Configuration
GCS_BUCKET = os.getenv("GCS_BUCKET", "footyai")
GCS_PROJECT = os.getenv("GCS_PROJECT", "learnagentspace")

def upload_to_gcs(local_path: str, gcs_path: str):
    """Upload a file to Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to gs://{GCS_BUCKET}/{gcs_path}")
    except Exception as e:
        print(f"Failed to upload {local_path} to GCS: {e}")

def save_model_to_gcs(model_state_dict, model_name: str):
    """Save model checkpoint to GCS."""
    local_path = f"/tmp/{model_name}"
    torch.save(model_state_dict, local_path)
    gcs_path = f"models/{model_name}"
    upload_to_gcs(local_path, gcs_path)
    return f"gs://{GCS_BUCKET}/{gcs_path}"

# Hyperparameters
SIGMA_MULTIPLIER = 0.5
SAMPLE = True  # Whether to sample from the policy or use the mean
BATCH_SIZE = 32  # Number of parallel environments to simulate
NUM_EPISODES = 10000
GAMMA = 0.99
RENDER_EVERY = NUM_EPISODES // 100
LEARNING_RATE = 1e-3

T = GAME_DURATION

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
    # discounts = gamma ** torch.arange(T, dtype=rewards_tensor.dtype, device=rewards_tensor.device).view(T, 1)
    discounts = DISCOUNTS

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
    # Optimized with einsum for better performance
    loss_a = -torch.einsum('tbi,tb->', log_probs_a, discounted_rewards_a) / log_probs_a.size(1)
    loss_b = -torch.einsum('tbi,tb->', log_probs_b, discounted_rewards_b) / log_probs_b.size(1)

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
            rewards_steps = rewards_steps[-1].cpu().detach().numpy()
            # print(rewards_steps)
            print(f"Episode {episode+1}/{num_episodes}, Avg Score: {avg_score.cpu().numpy()}, Avg Reward: {np.mean(rewards_steps)}, Avg Abs Reward: {np.mean([np.abs(r) for r in rewards_steps])}")
            
            # Save video locally then upload to GCS
            local_video_dir = f"/tmp/videos/{VIDEO_PREFIX}"
            os.makedirs(local_video_dir, exist_ok=True)
            local_video_path = f"{local_video_dir}/episode_{episode+1}.mp4"
            states_to_mp4(rendered_states, local_video_path)
            
            # Upload to GCS
            gcs_video_path = f"videos/{VIDEO_PREFIX}/episode_{episode+1}.mp4"
            upload_to_gcs(local_video_path, gcs_video_path)

    # Save the final model to GCS
    model_path = save_model_to_gcs(policy_net.state_dict(), f"{VIDEO_PREFIX}_final.pt")
    print(f"Training complete! Model saved to {model_path}")

def main():
    global DISCOUNTS
    
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Create discount factors tensor on the determined device
    DISCOUNTS = GAMMA ** torch.arange(T, dtype=torch.float32, device=device).view(T, 1)
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
