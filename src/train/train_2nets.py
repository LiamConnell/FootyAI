import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import List, Tuple
from rich import print

from src.soccer_env import SoccerEnv
from src.config import N_PLAYERS
from src.visualization import states_to_mp4
from src.policy_network import PolicyNetwork


def train_self_play(env: SoccerEnv, 
                    team_a_net: PolicyNetwork, team_b_net: PolicyNetwork, 
                    optimizer_a: optim.Optimizer, optimizer_b: optim.Optimizer,
                    num_episodes: int, render_every: int, gamma: float = 0.99):
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        log_probs_a, log_probs_b = [], []
        rewards_a, rewards_b = [], []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Team A policy
            mu_a, sigma_a = team_a_net(state_tensor)
            dist_a = Normal(mu_a, sigma_a)
            action_a = dist_a.sample()
            log_prob_a = dist_a.log_prob(action_a)

            # Team B policy
            mu_b, sigma_b = team_b_net(state_tensor)
            dist_b = Normal(mu_b, sigma_b)
            action_b = dist_b.sample()
            log_prob_b = dist_b.log_prob(action_b)

            # Combine actions
            action_combined = np.concatenate([
                action_a.detach().numpy(),
                action_b.detach().numpy()
            ])

            next_state, reward, done, _, info = env.step(action_combined)

            # Separate rewards
            rewards_a.append(reward)
            rewards_b.append(-reward)  # Negative for B to reflect zero-sum

            log_probs_a.append(log_prob_a)
            log_probs_b.append(log_prob_b)

            state = next_state

        # Update both teams separately
        update_policy(optimizer_a, log_probs_a, rewards_a, gamma)
        update_policy(optimizer_b, log_probs_b, rewards_b, gamma)

        if episode % render_every == 0:
            print(f"Episode {episode+1}/{num_episodes}, Score: {info.score}")
            states_to_mp4(env.states, f"videos/selfplay_episode_{episode}.mp4")


def update_policy(optimizer: optim.Optimizer, log_probs: List[torch.Tensor], rewards: List[float], gamma: float = 0.99):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)

    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_loss = [-log_prob.sum() * R for log_prob, R in zip(log_probs, discounted_rewards)]
    policy_loss = torch.stack(policy_loss).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

def main():
    env = SoccerEnv()

    input_dim = env.observation_space.shape[0]
    hidden_dim = 128
    output_dim = N_PLAYERS * 4

    team_a_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
    team_b_net = PolicyNetwork(input_dim, hidden_dim, output_dim)

    optimizer_a = optim.Adam(team_a_net.parameters(), lr=1e-4)
    optimizer_b = optim.Adam(team_b_net.parameters(), lr=1e-4)

    num_episodes = 1000
    gamma = 0.99

    train_self_play(env, team_a_net, team_b_net, optimizer_a, optimizer_b,
                    num_episodes, int(num_episodes / 10), gamma)

if __name__ == "__main__":
    main()

            