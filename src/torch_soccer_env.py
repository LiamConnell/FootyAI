import torch
import math
import uuid
from typing import List, Tuple

from src.game_state import Player, GameState, Ball
from src.config import (
    N_PLAYERS,
    MIN_KICKING_DISTANCE,
    GAME_DURATION, 
    FIELD_WIDTH,
    FIELD_HEIGHT,
    GOAL_HEIGHT,
    GAME_LENGTH,
    BALL_FRICTION,
    TEAM_A_START_POSITIONS,
    TEAM_B_START_POSITIONS,
    BALL_START_POSITION, 
    MAX_VELOCITY, 
    MAX_KICK_FORCE
)

CLOSENESS_REWARD_MULTIPLIER = .3
KICK_MULTIPLIER = .5

class TorchSoccerEnv:
    def __init__(self, batch_size: int, device: str = "cpu"):
        """
        Initialize the soccer environment.
        
        Args:
            batch_size: Number of parallel environments to simulate
            n_players: Number of players per team
            device: Device to run computations on ("cpu" or "cuda")
        """
        
        # Environment setup
        self.batch_size = batch_size
        self.device = device
        
        # Initialize state tensors
        self.ball_position = None
        self.ball_velocity = None
        self.team1_positions = None
        self.team2_positions = None
        self.score = None
        self.game_time = None

        self.prev_avg_dist_team1 = None
        self.prev_avg_dist_team2 = None

        self.reset()
        
    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            observation: Tensor containing the flattened state
        """
        # Initialize the ball at the center with zero velocity
        self.ball_position = torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], 
                                         device=self.device).repeat(self.batch_size, 1)
        self.ball_velocity = torch.zeros((self.batch_size, 2), device=self.device)
        
        # Initialize players with random positions (but in their respective field halves)
        self.team1_positions = torch.zeros((self.batch_size,N_PLAYERS, 2), device=self.device)
        self.team2_positions = torch.zeros((self.batch_size,N_PLAYERS, 2), device=self.device)
        
        if False:
            # Team 1 starts on the left half
            self.team1_positions[:, :, 0] = torch.rand((self.batch_size,N_PLAYERS), device=self.device) * (FIELD_WIDTH / 2 - 5) + 5
            self.team1_positions[:, :, 1] = torch.rand((self.batch_size,N_PLAYERS), device=self.device) * (FIELD_HEIGHT - 10) + 5
            
            # Team 2 starts on the right half
            self.team2_positions[:, :, 0] = torch.rand((self.batch_size,N_PLAYERS), device=self.device) * (FIELD_WIDTH / 2 - 5) + (FIELD_WIDTH / 2)
            self.team2_positions[:, :, 1] = torch.rand((self.batch_size,N_PLAYERS), device=self.device) * (FIELD_HEIGHT - 10) + 5
        else:
            # Team 1 starts anywhere
            self.team1_positions[:, :, 0] = torch.rand((self.batch_size,N_PLAYERS), device=self.device) * (FIELD_WIDTH)
            self.team1_positions[:, :, 1] = torch.rand((self.batch_size,N_PLAYERS), device=self.device) * (FIELD_HEIGHT)
            
            # Team 2 starts anywhere
            self.team2_positions[:, :, 0] = torch.rand((self.batch_size,N_PLAYERS), device=self.device) * (FIELD_WIDTH)
            self.team2_positions[:, :, 1] = torch.rand((self.batch_size,N_PLAYERS), device=self.device) * (FIELD_HEIGHT)

        # Initialize previous average distances
        team1_to_ball = torch.norm(self.team1_positions - self.ball_position.unsqueeze(1), dim=2)
        team2_to_ball = torch.norm(self.team2_positions - self.ball_position.unsqueeze(1), dim=2)
        self.prev_avg_dist_team1 = torch.mean(team1_to_ball, dim=1)
        self.prev_avg_dist_team2 = torch.mean(team2_to_ball, dim=1)
        
        # Initialize score to zeros
        self.score = torch.zeros((self.batch_size, 2), device=self.device)
        
        # Initialize game time
        self.game_time = torch.zeros(self.batch_size, device=self.device)
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, actions: torch.Tensor):
        """
        Take a step in the environment based on the actions of all players.
        
        Args:
            actions: Tensor of shape (batch_size, n_players*2, 4) where the last dimension
                     contains [move_vel_x, move_vel_y, kick_vel_x, kick_vel_y] for each player
                     
        Returns:
            observation: Tensor containing the flattened state
            reward: Reward for this step (1 for team 1 scoring, -1 for team 2 scoring, 0 otherwise)
            done: Boolean indicating if the game is over
            info: Additional information (empty dictionary for now)
        """
        # # Extract player movement and kick velocities
        # actions = torch.clamp(actions, -2.0, 2.0)

        team1_move_vel = actions[:, :N_PLAYERS, :2]
        team1_kick_vel = actions[:, :N_PLAYERS, 2:4]
        team2_move_vel = actions[:,N_PLAYERS:, :2]
        team2_kick_vel = actions[:,N_PLAYERS:, 2:4]
        
        # Move players
        self.team1_positions = self.team1_positions + team1_move_vel * MAX_VELOCITY
        self.team2_positions = self.team2_positions + team2_move_vel * MAX_VELOCITY
        
        # Constrain players to field boundaries
        self.team1_positions[:, :, 0] = torch.clamp(self.team1_positions[:, :, 0], 0,FIELD_WIDTH)
        self.team1_positions[:, :, 1] = torch.clamp(self.team1_positions[:, :, 1], 0,FIELD_HEIGHT)
        self.team2_positions[:, :, 0] = torch.clamp(self.team2_positions[:, :, 0], 0,FIELD_WIDTH)
        self.team2_positions[:, :, 1] = torch.clamp(self.team2_positions[:, :, 1], 0,FIELD_HEIGHT)
        
        # Calculate distances from all players to the ball
        ball_expanded = self.ball_position.unsqueeze(1)  # [batch_size, 1, 2]
        
        team1_to_ball = torch.norm(self.team1_positions - ball_expanded, dim=2)  # [batch_size, n_players]
        team2_to_ball = torch.norm(self.team2_positions - ball_expanded, dim=2)  # [batch_size, n_players]
        
        # Find which players are within kicking distance
        team1_can_kick = team1_to_ball <MIN_KICKING_DISTANCE  # [batch_size, n_players]
        team2_can_kick = team2_to_ball <MIN_KICKING_DISTANCE  # [batch_size, n_players]
        
        # Calculate new ball velocity based on kicks
        new_ball_velocity = torch.zeros_like(self.ball_velocity)
        
        # Sum up kick contributions for team 1
        team1_kick_mask = team1_can_kick.float().unsqueeze(-1)  # [batch_size, n_players, 1]
        team1_total_kicks = torch.sum(team1_kick_mask * team1_kick_vel, dim=1)  # [batch_size, 2]
        team1_kickers_count = torch.sum(team1_kick_mask, dim=(1, 2))  # [batch_size]
        
        # Sum up kick contributions for team 2
        team2_kick_mask = team2_can_kick.float().unsqueeze(-1)  # [batch_size, n_players, 1]
        team2_total_kicks = torch.sum(team2_kick_mask * team2_kick_vel, dim=1)  # [batch_size, 2]
        team2_kickers_count = torch.sum(team2_kick_mask, dim=(1, 2))  # [batch_size]
        
        # Combine kicks from both teams
        total_kicks = team1_total_kicks + team2_total_kicks
        total_kickers = team1_kickers_count + team2_kickers_count
        
        # Update ball velocity only if there are kickers
        kicking_mask = (total_kickers > 0).float().unsqueeze(-1)  # [batch_size, 1]
        safe_kickers = torch.clamp(total_kickers.unsqueeze(-1), min=1.0)  # Avoid division by zero
        
        # If any player is kicking, update velocity to the average of kicks
        averaged_kicks = total_kicks / safe_kickers
        new_ball_velocity = kicking_mask * averaged_kicks * MAX_KICK_FORCE + (1 - kicking_mask) * self.ball_velocity * BALL_FRICTION
        
        # If no player is kicking, decay the current velocity
        self.ball_velocity = new_ball_velocity
        
        # Move the ball according to its velocity
        old_ball_position = self.ball_position.clone()
        self.ball_position = self.ball_position + self.ball_velocity
        
        # Handle ball out of bounds - bounce off walls
        # X-boundaries (sides)
        x_out_left = self.ball_position[:, 0] < 0
        x_out_right = self.ball_position[:, 0] >FIELD_WIDTH
        
        # Y-boundaries (top and bottom)
        y_out_top = self.ball_position[:, 1] < 0
        y_out_bottom = self.ball_position[:, 1] >FIELD_HEIGHT
        
        # Bounce ball off walls by reversing velocity component and constraining position
        self.ball_velocity[x_out_left | x_out_right, 0] *= -1
        self.ball_velocity[y_out_top | y_out_bottom, 1] *= -1
        
        self.ball_position[:, 0] = torch.clamp(self.ball_position[:, 0], 0,FIELD_WIDTH)
        self.ball_position[:, 1] = torch.clamp(self.ball_position[:, 1], 0,FIELD_HEIGHT)
        
        # Left goal (team 2 scores)
        goal_y_range = torch.logical_and(
            self.ball_position[:, 1] >= (FIELD_HEIGHT -GOAL_HEIGHT) / 2,
            self.ball_position[:, 1] <= (FIELD_HEIGHT +GOAL_HEIGHT) / 2
        )
        
        # Compute scoring events for both teams
        team2_scored = torch.logical_and(x_out_left, goal_y_range)
        team1_scored = torch.logical_and(x_out_right, goal_y_range)

        # Update score
        self.score[:, 1] += team2_scored.float()
        self.score[:, 0] += team1_scored.float()

        # Calculate reward
        reward = torch.zeros(self.batch_size, device=self.device)

        # Add goal rewards
        reward += team1_scored.float()    # +1 when team 1 scores
        reward -= team2_scored.float()    # -1 when team 2 scores

        # Create a mask for instances where a goal has been scored (by either team)
        scored_mask = torch.logical_or(team1_scored, team2_scored)  # Boolean tensor of shape [batch_size]
        # Invert mask: 1 if no team scored, 0 if either scored
        non_scoring_mask = (~scored_mask).float()

        # Compute average distance from the ball for both teams
        avg_dist_team1 = torch.mean(team1_to_ball, dim=1)  # Shape: [batch_size]
        avg_dist_team2 = torch.mean(team2_to_ball, dim=1)  # Shape: [batch_size]

        # Compute change in average distance (positive if team moves closer to the ball)
        delta_dist_team1 = self.prev_avg_dist_team1 - avg_dist_team1
        delta_dist_team2 = self.prev_avg_dist_team2 - avg_dist_team2

        # Shaping reward: team1 gets a positive value when moving closer,
        # team2 gets a negative value when moving closer
        distance_shaping_reward = delta_dist_team1 - delta_dist_team2

        # Apply distance shaping reward only if no team scored in that instance
        reward += distance_shaping_reward * CLOSENESS_REWARD_MULTIPLIER * non_scoring_mask

        # Reward team kicks
        reward += ((team1_kickers_count>0).float() - (team2_kickers_count>0).float()) * KICK_MULTIPLIER
        




        # Update previous distances for the next step
        self.prev_avg_dist_team1 = avg_dist_team1
        self.prev_avg_dist_team2 = avg_dist_team2
        
        # Reset ball to center after goals
        scored = team1_scored | team2_scored
        
        # Reset ball position and velocity where goals were scored
        self.ball_position[scored, 0] =FIELD_WIDTH / 2
        self.ball_position[scored, 1] =FIELD_HEIGHT / 2
        self.ball_velocity[scored] = 0
        
        # Increment game time
        self.game_time += 1
        
        # Check if game is done
        done = self.game_time >=GAME_LENGTH
        
        # Return observation, reward, done flag
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """
        Construct and return the observation tensor.
        
        Returns:
            observation: Tensor of shape [batch_size, observation_size]
        """
        # Concatenate all state information into a single tensor
        obs_components = [
            self.ball_position,                                      # [batch_size, 2]
            self.ball_velocity,                                       # [batch_size, 2]
            self.team1_positions.reshape(self.batch_size, -1),       # [batch_size, n_players*2]
            self.team2_positions.reshape(self.batch_size, -1),       # [batch_size, n_players*2]
            self.score,                                               # [batch_size, 2]
            self.game_time.unsqueeze(-1)                              # [batch_size, 1]
        ]
        
        # Concatenate along the last dimension
        observation = torch.cat(obs_components, dim=1)
        
        return observation
    
    def to_pydantic(self, batch_idx=0):
        """
        Convert the current state to a Pydantic GameState object.
        
        Args:
            batch_idx: Index of the batch to convert to a GameState
            
        Returns:
            GameState: A Pydantic model representation of the current state
        """
        # Import necessary Pydantic models here to avoid circular imports
        
        # Extract the data for the specified batch index
        ball_pos = (float(self.ball_position[batch_idx, 0].item()), float(self.ball_position[batch_idx, 1].item()))
        ball_vel = (float(self.ball_velocity[batch_idx, 0].item()), float(self.ball_velocity[batch_idx, 1].item()))
        
        # Create team A players
        team_a = []
        for i in range(N_PLAYERS):
            pos = (float(self.team1_positions[batch_idx, i, 0].item()), 
                   float(self.team1_positions[batch_idx, i, 1].item()))
            # We don't explicitly track player velocities in our implementation
            # so we're setting them to zero here
            vel = (0.0, 0.0)
            team_a.append(Player(position=pos, velocity=vel))
        
        # Create team B players
        team_b = []
        for i in range(N_PLAYERS):
            pos = (float(self.team2_positions[batch_idx, i, 0].item()), 
                   float(self.team2_positions[batch_idx, i, 1].item()))
            vel = (0.0, 0.0)
            team_b.append(Player(position=pos, velocity=vel))
        
        # Create score tuple
        score = (int(self.score[batch_idx, 0].item()), int(self.score[batch_idx, 1].item()))
        
        # Calculate time remaining
        time_remaining = float(GAME_LENGTH - self.game_time[batch_idx].item())
        
        # Create and return the GameState
        return GameState(
            team_a=team_a,
            team_b=team_b,
            ball=Ball(position=ball_pos, velocity=ball_vel),
            score=score,
            time_remaining=time_remaining
        )


# Example usage
if __name__ == "__main__":
    from tqdm import tqdm
    from src.visualization import states_to_mp4
    
    # Create an environment with batch size of 2 and 3 players per team
    env = TorchSoccerEnv(batch_size=2, n_players=3, device="cpu")
    
    # Reset the environment
    obs = env.reset()
    
    # Run a few steps with random actions
    states = []
    for i in tqdm(range(100)):
        # Random actions: [move_x, move_y, kick_x, kick_y] for each player on both teams
        random_actions = torch.randn(2, 6, 4) * 2.0  # Scale for more movement
        
        # Take a step
        obs, reward, done, _ = env.step(random_actions)
        
        states.append(env.to_pydantic(batch_idx=0))

    states_to_mp4(states, "test_game.mp4")
