import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from .policy_network import SoccerPolicyNetwork
from ..game_state import GameState, PlayerAction
from ..visualization import SoccerVisualizer
from typing import List, Tuple

class PPOTrainer:
    def __init__(self, 
                 input_size=44,
                 hidden_size=128,
                 output_size=20,
                 lr=3e-4,
                 gamma=0.99,
                 epsilon=0.2,
                 batch_size=64,
                 buffer_size=10000):
        
        self.policy_net = SoccerPolicyNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("visualizations", exist_ok=True)
        
    def _get_state_vector(self, game_state: GameState) -> np.ndarray:
        """Convert game state to a vector for the neural network."""
        state = []
        
        # Add ball position and velocity
        state.extend(game_state.ball.position)
        state.extend(game_state.ball.velocity)
        
        # Add team A players' positions and velocities
        for player in game_state.team_a:
            state.extend(player.position)
            state.extend(player.velocity)
            
        # Add team B players' positions and velocities
        for player in game_state.team_b:
            state.extend(player.position)
            state.extend(player.velocity)
            
        return np.array(state, dtype=np.float32)
        
    def _convert_to_player_actions(self, action_array: np.ndarray) -> List[PlayerAction]:
        """Convert numpy array to list of PlayerAction objects."""
        actions = []
        for i in range(0, len(action_array), 2):
            actions.append(PlayerAction(
                vx=float(action_array[i]),
                vy=float(action_array[i+1]),
                kx=0.0,  # No kicking in this version
                ky=0.0   # No kicking in this version
            ))
        return actions
        
    def collect_experience(self, num_episodes=10, record_game=False) -> List[GameState]:
        """Collect experience and optionally record the game states."""
        game_states = []
        for _ in range(num_episodes):
            game_state = GameState.create_default()
            if record_game:
                game_states.append(game_state)
            state = self._get_state_vector(game_state)
            done = False
            episode_reward = 0
            
            while not done:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Get action from policy
                with torch.no_grad():
                    action = self.policy_net.get_action(state_tensor)
                
                # Take action in environment
                # Split action into team A and B actions
                action_np = action.numpy()[0]  # Remove batch dimension
                team_a_actions = self._convert_to_player_actions(action_np[:10])  # First 10 values for team A
                team_b_actions = self._convert_to_player_actions(action_np[10:])  # Last 10 values for team B
                
                # Update game state
                next_game_state = game_state.update(team_a_actions, team_b_actions)
                if record_game:
                    game_states.append(next_game_state)
                next_state = self._get_state_vector(next_game_state)
                
                # Calculate reward (difference in score)
                reward = float(next_game_state.score[0] - game_state.score[0])
                
                # Check if game is done (time up or score limit reached)
                done = next_game_state.time_remaining <= 0 or max(next_game_state.score) >= 5
                
                # Store experience
                self.buffer.append((state, action_np, reward, next_state, done))
                
                state = next_state
                game_state = next_game_state
                episode_reward += reward
                
            print(f"Episode reward: {episode_reward}")
        return game_states if record_game else None
    
    def train(self, num_epochs=10):
        if len(self.buffer) < self.batch_size:
            return
        
        for _ in range(num_epochs):
            # Sample batch
            batch = random.sample(self.buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(states))  # Shape: [batch_size, input_size]
            actions = torch.FloatTensor(np.array(actions))  # Shape: [batch_size, output_size]
            rewards = torch.FloatTensor(np.array(rewards))  # Shape: [batch_size]
            next_states = torch.FloatTensor(np.array(next_states))  # Shape: [batch_size, input_size]
            dones = torch.FloatTensor(np.array(dones))  # Shape: [batch_size]
            
            # Get current policy and value
            current_policy, current_value = self.policy_net(states)  # Shapes: [batch_size, output_size], [batch_size, 1]
            
            # Get next value
            with torch.no_grad():
                _, next_value = self.policy_net(next_states)  # Shape: [batch_size, 1]
            
            # Calculate advantages
            advantages = rewards.unsqueeze(1) + self.gamma * next_value * (1 - dones.unsqueeze(1)) - current_value
            
            # Calculate policy loss
            policy_loss = -torch.min(
                advantages * torch.exp(current_policy - actions),
                advantages * torch.clamp(torch.exp(current_policy - actions), 
                                       1 - self.epsilon, 
                                       1 + self.epsilon)
            ).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(current_value, rewards.unsqueeze(1) + self.gamma * next_value * (1 - dones.unsqueeze(1)))
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))

def main():
    trainer = PPOTrainer()
    visualizer = SoccerVisualizer()
    
    # Training loop
    for episode in range(1000):
        print(f"Episode {episode}")
        
        # Collect experience and optionally record game states
        record_game = (episode % 100 == 0)
        game_states = trainer.collect_experience(num_episodes=1, record_game=record_game)
        
        # Train on collected experience
        trainer.train(num_epochs=10)
        
        # Save model and visualization periodically
        if episode % 100 == 0:
            # # Save model
            # model_path = f"models/soccer_policy_{episode}.pt"
            # trainer.save_model(model_path)
            
            # Save visualization
            if game_states:
                vis_path = f"visualizations/game_{episode}.mp4"
                visualizer.save_game_sequence(game_states, vis_path, show=False)
                print(f"Saved visualization to {vis_path}")

if __name__ == "__main__":
    main() 