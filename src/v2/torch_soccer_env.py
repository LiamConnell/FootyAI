import torch
import math
import uuid
from typing import List, Tuple
from src.game_state import Ball, GameState, Player

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
    MAX_KICK_FORCE,
    RANDOM_INITIALIZATION
)

CLOSENESS_REWARD_MULTIPLIER = 0.3
KICK_MULTIPLIER = 0.5

class TorchSoccerEnv:
    def __init__(self, batch_size: int, device: str = "cpu"):
        """
        Initialize the soccer environment.

        Args:
            batch_size: Number of parallel environments to simulate
            device: Device to run computations on ("cpu" or "cuda")
        """
        self.batch_size = batch_size
        self.device = device
        # print(self.batch_size)

        # Pre-allocate state tensors
        self.ball_position = torch.empty((self.batch_size, 2), device=self.device)
        self.ball_velocity = torch.empty((self.batch_size, 2), device=self.device)
        self.team1_positions = torch.empty((self.batch_size, N_PLAYERS, 2), device=self.device)
        self.team2_positions = torch.empty((self.batch_size, N_PLAYERS, 2), device=self.device)
        self.score = torch.zeros((self.batch_size, 2), device=self.device)
        self.game_time = torch.zeros(self.batch_size, device=self.device)

        # Pre-allocate tensors to store previous distances
        self.prev_avg_dist_team1 = torch.empty(self.batch_size, device=self.device)
        self.prev_avg_dist_team2 = torch.empty(self.batch_size, device=self.device)

        # Initialize players using proper soccer starting positions
        self.team1_starts = torch.tensor(TEAM_A_START_POSITIONS, device=self.device, dtype=torch.float32)
        self.team2_starts = torch.tensor(TEAM_B_START_POSITIONS, device=self.device, dtype=torch.float32)

        # Run an initial reset so that state tensors hold a valid initial state.
        self.reset()

    def _generate_random_positions(self):
        """
        Generate random starting positions for both teams.
        Team 1 gets left half, Team 2 gets right half of the field.
        """
        # Team 1: left half of field (x: 5 to FIELD_WIDTH/2 - 5)
        team1_x = torch.rand(self.batch_size, N_PLAYERS, device=self.device) * (FIELD_WIDTH/2 - 10) + 5
        team1_y = torch.rand(self.batch_size, N_PLAYERS, device=self.device) * (FIELD_HEIGHT - 10) + 5
        self.team1_positions[:, :, 0] = team1_x
        self.team1_positions[:, :, 1] = team1_y
        
        # Team 2: right half of field (x: FIELD_WIDTH/2 + 5 to FIELD_WIDTH - 5)
        team2_x = torch.rand(self.batch_size, N_PLAYERS, device=self.device) * (FIELD_WIDTH/2 - 10) + FIELD_WIDTH/2 + 5
        team2_y = torch.rand(self.batch_size, N_PLAYERS, device=self.device) * (FIELD_HEIGHT - 10) + 5
        self.team2_positions[:, :, 0] = team2_x
        self.team2_positions[:, :, 1] = team2_y

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            observation: Tensor containing the flattened state
        """
        # Set ball at the center and zero its velocity
        center_ = torch.tensor([FIELD_WIDTH / 2, FIELD_HEIGHT / 2], device=self.device)
        self.center = center_.repeat(self.batch_size, 1)
        self.ball_position.copy_(self.center)
        self.ball_velocity.zero_()

        # Set starting positions for all batches
        if RANDOM_INITIALIZATION:
            self._generate_random_positions()
        else:
            self.team1_positions.copy_(self.team1_starts.unsqueeze(0).expand(self.batch_size, -1, -1))
            self.team2_positions.copy_(self.team2_starts.unsqueeze(0).expand(self.batch_size, -1, -1))

        # Compute previous average distances from ball to team players.
        # These calculations occur on the pre-existing tensors.
        # Optimized with einsum for better performance
        diff1 = self.team1_positions - self.ball_position.unsqueeze(1)
        team1_to_ball = torch.sqrt(torch.einsum('bni,bni->bn', diff1, diff1))
        diff2 = self.team2_positions - self.ball_position.unsqueeze(1)
        team2_to_ball = torch.sqrt(torch.einsum('bni,bni->bn', diff2, diff2))
        self.prev_avg_dist_team1.copy_(torch.einsum('bn->b', team1_to_ball) / N_PLAYERS)
        self.prev_avg_dist_team2.copy_(torch.einsum('bn->b', team2_to_ball) / N_PLAYERS)

        # Reset score and game time using in-place operations.
        self.score.zero_()
        self.game_time.zero_()

        # Return the initial observation
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
        # Extract movement and kicking velocities for both teams
        team1_move_vel = actions[:, :N_PLAYERS, :2]
        team1_kick_vel = actions[:, :N_PLAYERS, 2:4]
        team2_move_vel = actions[:, N_PLAYERS:, :2]
        team2_kick_vel = actions[:, N_PLAYERS:, 2:4]

        # Update player positions and clamp them to field boundaries
        self.team1_positions.add_(team1_move_vel * MAX_VELOCITY)
        self.team2_positions.add_(team2_move_vel * MAX_VELOCITY)

        self.team1_positions[:, :, 0].clamp_(0, FIELD_WIDTH)
        self.team1_positions[:, :, 1].clamp_(0, FIELD_HEIGHT)
        self.team2_positions[:, :, 0].clamp_(0, FIELD_WIDTH)
        self.team2_positions[:, :, 1].clamp_(0, FIELD_HEIGHT)

        # Calculate distances from players to ball
        # Optimized with einsum for better performance
        ball_expanded = self.ball_position.unsqueeze(1)  # [batch_size, 1, 2]
        diff1 = self.team1_positions - ball_expanded
        team1_to_ball = torch.sqrt(torch.einsum('bni,bni->bn', diff1, diff1))
        diff2 = self.team2_positions - ball_expanded
        team2_to_ball = torch.sqrt(torch.einsum('bni,bni->bn', diff2, diff2))

        # Determine players within kicking distance
        team1_can_kick = team1_to_ball < MIN_KICKING_DISTANCE
        team2_can_kick = team2_to_ball < MIN_KICKING_DISTANCE

        # Calculate new ball velocity based on kicks
        # Optimized with einsum for better performance
        team1_kick_mask = team1_can_kick.float().unsqueeze(-1)
        team1_total_kicks = torch.einsum('bni,bni->bi', team1_kick_mask, team1_kick_vel)
        team1_kickers_count = torch.einsum('bni->b', team1_kick_mask)

        team2_kick_mask = team2_can_kick.float().unsqueeze(-1)
        team2_total_kicks = torch.einsum('bni,bni->bi', team2_kick_mask, team2_kick_vel)
        team2_kickers_count = torch.einsum('bni->b', team2_kick_mask)

        total_kicks = team1_total_kicks + team2_total_kicks
        total_kickers = team1_kickers_count + team2_kickers_count

        kicking_mask = (total_kickers > 0).float().unsqueeze(-1)
        safe_kickers = torch.clamp(total_kickers.unsqueeze(-1), min=1.0)
        averaged_kicks = total_kicks / safe_kickers

        new_ball_velocity = kicking_mask * averaged_kicks * MAX_KICK_FORCE + \
                            (1 - kicking_mask) * self.ball_velocity * BALL_FRICTION
        self.ball_velocity.copy_(new_ball_velocity)

        # Move the ball according to its velocity.
        self.ball_position.add_(self.ball_velocity)

        # Bounce off walls
        x_out_left = self.ball_position[:, 0] < 0
        x_out_right = self.ball_position[:, 0] > FIELD_WIDTH
        y_out_top = self.ball_position[:, 1] < 0
        y_out_bottom = self.ball_position[:, 1] > FIELD_HEIGHT

        # self.ball_velocity[x_out_left | x_out_right, 0].mul_(-1)
        # self.ball_velocity[y_out_top | y_out_bottom, 1].mul_(-1)
        self.ball_velocity[x_out_left | x_out_right, 0] = -1 * self.ball_velocity[x_out_left | x_out_right, 0]
        self.ball_velocity[y_out_top | y_out_bottom, 1] = -1 * self.ball_velocity[y_out_top | y_out_bottom, 1].mul_(-1)

        self.ball_position[:, 0].clamp_(0, FIELD_WIDTH)
        self.ball_position[:, 1].clamp_(0, FIELD_HEIGHT)

        # Check for goals.
        goal_y_range = torch.logical_and(
            self.ball_position[:, 1] >= (FIELD_HEIGHT - GOAL_HEIGHT) / 2,
            self.ball_position[:, 1] <= (FIELD_HEIGHT + GOAL_HEIGHT) / 2
        )
        team2_scored = torch.logical_and(x_out_left, goal_y_range)
        team1_scored = torch.logical_and(x_out_right, goal_y_range)

        # Update scores and compute reward.
        self.score[:, 1].add_(team2_scored.float())
        self.score[:, 0].add_(team1_scored.float())

        reward = torch.zeros(self.batch_size, device=self.device)
        reward.add_(team1_scored.float())
        reward.sub_(team2_scored.float())

        scored_mask = torch.logical_or(team1_scored, team2_scored)
        non_scoring_mask = (~scored_mask).float()

        avg_dist_team1 = torch.einsum('bn->b', team1_to_ball) / N_PLAYERS
        avg_dist_team2 = torch.einsum('bn->b', team2_to_ball) / N_PLAYERS
        delta_dist_team1 = self.prev_avg_dist_team1 - avg_dist_team1
        delta_dist_team2 = self.prev_avg_dist_team2 - avg_dist_team2
        distance_shaping_reward = delta_dist_team1 - delta_dist_team2
        reward.add_(distance_shaping_reward * CLOSENESS_REWARD_MULTIPLIER * non_scoring_mask)

        reward.add_(((team1_kickers_count > 0).float() - (team2_kickers_count > 0).float()) * KICK_MULTIPLIER)

        # Update previous distances
        self.prev_avg_dist_team1.copy_(avg_dist_team1)
        self.prev_avg_dist_team2.copy_(avg_dist_team2)

        # Reset ball if a goal is scored.
        scored = team1_scored | team2_scored
        reset_mask = scored.unsqueeze(-1)
        self.ball_position = torch.where(reset_mask, self.center, self.ball_position)
        self.ball_velocity = torch.where(reset_mask, torch.zeros_like(self.ball_velocity), self.ball_velocity)

        # Update game time and determine if the game has finished.
        self.game_time.add_(1)
        done = self.game_time >= GAME_LENGTH
        
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """
        Construct and return the observation tensor with normalization.
        
        Returns:
            observation: Tensor of shape [batch_size, observation_size] with values in [-1, 1]
        """
        # Normalize positions to [-1, 1] range
        norm_team1_pos = (self.team1_positions / torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], device=self.device)) - 1
        norm_team2_pos = (self.team2_positions / torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], device=self.device)) - 1
        norm_ball_pos = (self.ball_position / torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], device=self.device)) - 1
        
        # Normalize velocities (assume max velocity is reasonable for normalization)
        max_vel = max(MAX_VELOCITY, MAX_KICK_FORCE)
        norm_ball_vel = self.ball_velocity / max_vel
        
        # Normalize scores (assuming max reasonable score difference is 10)
        norm_score = self.score / 10.0
        
        # Normalize game time to [0, 1]
        norm_time = self.game_time.unsqueeze(-1) / GAME_LENGTH
        
        obs_components = [
            norm_team1_pos.reshape(self.batch_size, -1),            # [batch_size, n_players*2]
            norm_team2_pos.reshape(self.batch_size, -1),            # [batch_size, n_players*2]
            norm_ball_pos,                                          # [batch_size, 2]
            norm_ball_vel,                                          # [batch_size, 2]
            norm_score,                                             # [batch_size, 2]
            norm_time                                               # [batch_size, 1]
        ]
        return torch.cat(obs_components, dim=1)
    
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
