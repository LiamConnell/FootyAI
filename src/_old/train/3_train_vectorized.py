import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import List
from rich import print
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm

from src.soccer_env import SoccerEnv
from src.config import N_PLAYERS
from src.visualization import states_to_mp4
from src.policy_network import PolicyNetwork

from datetime import datetime


VIDEO_PREFIX = f"vectorized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

SIGMA_MUTLIPLIER = .1

SAMPLE = True


def make_env():
    return SoccerEnv()


def train_self_play(policy_net: PolicyNetwork,
                    optimizer: optim.Optimizer,
                    num_envs: int,
                    num_episodes: int, render_every: int,
                    device: str, gamma: float = 0.99):

    envs = SyncVectorEnv([make_env for _ in range(num_envs)])

    for episode in tqdm(range(num_episodes)):
        states, _ = envs.reset()
        dones = [False] * num_envs

        log_probs, rewards = [[] for _ in range(num_envs)], [[] for _ in range(num_envs)]

        while not all(dones):
            states_tensor = torch.tensor(states, dtype=torch.float32, device=device)

            team_a_inputs = torch.cat([states_tensor, torch.tensor([[1, 0]] * num_envs, device=device)], dim=1)
            mu_a, sigma_a = policy_net(team_a_inputs)
            mu_a = torch.tanh(mu_a)
            if SAMPLE: 
                dist_a = Normal(mu_a, sigma_a*SIGMA_MUTLIPLIER)
                actions_a = dist_a.sample()
                log_probs_a = dist_a.log_prob(actions_a)
            else:
                actions_a = mu_a
                log_probs_a = torch.zeros_like(actions_a)
                

            team_b_inputs = torch.cat([states_tensor, torch.tensor([[0, 1]] * num_envs, device=device)], dim=1)
            mu_b, sigma_b = policy_net(team_b_inputs)
            mu_b = torch.tanh(mu_b)
            if SAMPLE:
                dist_b = Normal(mu_b, sigma_b*SIGMA_MUTLIPLIER)
                actions_b = dist_b.sample()
                log_probs_b = dist_b.log_prob(actions_b)
            else:
                actions_b = mu_b
                log_probs_b = torch.zeros_like(actions_b)
                

            actions_combined = np.concatenate([actions_a.cpu().detach().numpy(), actions_b.cpu().detach().numpy()], axis=1)

            next_states, rewards_step, dones, _, infos = envs.step(actions_combined)

            for i in range(num_envs):
                if not dones[i]:
                    log_probs[i].append(log_probs_a[i])
                    rewards[i].append(rewards_step[i])

                    log_probs[i].append(log_probs_b[i])
                    rewards[i].append(-rewards_step[i])

            states = next_states

        update_policy(optimizer, log_probs, rewards, gamma)


        if episode % render_every == -1 % render_every:
            avg_score = np.mean([env.game_state.score for env in envs.envs], axis=0)
            print(f"Episode {episode+1}/{num_episodes}, Score: {avg_score}")
            os.makedirs(f"videos/{VIDEO_PREFIX}", exist_ok=True)
            states_to_mp4(envs.envs[0].states, f"videos/{VIDEO_PREFIX}/{episode+1}.mp4")


def update_policy(optimizer: optim.Optimizer, log_probs: List[List[torch.Tensor]], rewards: List[List[float]], gamma: float = 0.99):
    all_discounted_rewards = []
    all_log_probs = []

    for env_log_probs, env_rewards in zip(log_probs, rewards):
        discounted_rewards = []
        R = 0
        for r in reversed(env_rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        all_discounted_rewards.extend(discounted_rewards)
        all_log_probs.extend(env_log_probs)

    policy_loss = [-log_prob.sum() * R for log_prob, R in zip(all_log_probs, all_discounted_rewards)]
    policy_loss = torch.stack(policy_loss).sum()

    print(policy_loss)

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"

    print("Using device:", device)
    

    input_dim = SoccerEnv().observation_space.shape[0] + 2
    hidden_dim = 128
    output_dim = N_PLAYERS * 4

    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    num_envs = 64
    num_episodes = 5000
    gamma = 0.99

    train_self_play(policy_net, optimizer, num_envs,
                    num_episodes, int(num_episodes / 10), device, gamma)


if __name__ == "__main__":
    main()
