import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import List
from rich import print

from src.soccer_env import SoccerEnv
from src.config import N_PLAYERS
from src.visualization import states_to_mp4
from src.policy_network import PolicyNetwork


def train_self_play(env: SoccerEnv, 
                    policy_net: PolicyNetwork,
                    optimizer: optim.Optimizer,
                    num_episodes: int, render_every: int, gamma: float = 0.99):
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        log_probs, rewards = [], []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Team A policy
            team_a_input = torch.cat([state_tensor, torch.tensor([1, 0])])
            mu_a, sigma_a = policy_net(team_a_input)
            dist_a = Normal(mu_a, sigma_a)
            action_a = dist_a.sample()
            log_prob_a = dist_a.log_prob(action_a)

            # Team B policy
            team_b_input = torch.cat([state_tensor, torch.tensor([0, 1])])
            mu_b, sigma_b = policy_net(team_b_input)
            dist_b = Normal(mu_b, sigma_b)
            action_b = dist_b.sample()
            log_prob_b = dist_b.log_prob(action_b)

            # Combine actions
            action_combined = np.concatenate([
                action_a.detach().numpy(),
                action_b.detach().numpy()
            ])

            next_state, reward, done, _, info = env.step(action_combined)

            log_probs.append(log_prob_a)
            rewards.append(reward)

            log_probs.append(log_prob_b)
            rewards.append(-reward)

            state = next_state

        update_policy(optimizer, log_probs, rewards, gamma)

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

    input_dim = env.observation_space.shape[0] + 2  # Additional input for one-hot team encoding
    hidden_dim = 128
    output_dim = N_PLAYERS * 4

    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    num_episodes = 1000
    gamma = 0.99

    train_self_play(env, policy_net, optimizer,
                    num_episodes, int(num_episodes / 10), gamma)


if __name__ == "__main__":
    main()