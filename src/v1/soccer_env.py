import numpy as np
from typing import List, Tuple, Dict, Any, ClassVar
import gymnasium as gym
from gymnasium import spaces

from src.game_state import GameState, PlayerAction
from src.visualization import states_to_mp4
from src.config import (
    N_PLAYERS,
    FIELD_WIDTH,
    FIELD_HEIGHT,
    MAX_VELOCITY,
    MAX_KICK_FORCE,
    GAME_DURATION
)

SPARSE_REWARDS = True
CLOSENESS_REWARD_MULTIPLIER = 1


class SoccerEnv(gym.Env):
    """A soccer environment for reinforcement learning."""
    
    metadata: ClassVar[Dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: str = None):
        super().__init__()
        
        self.render_mode = render_mode
        # self.visualizer = SoccerVisualizer() if render_mode else None
        self.game_state = GameState.create_default()
        self.states: list[GameState] = []

        self.sparse_rewards = SPARSE_REWARDS
        
        # Define action space: [vx, vy, kx, ky] for each player
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(N_PLAYERS * 2, 4),  # 2 teams * N_PLAYERS players * 4 actions
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=((N_PLAYERS * 2 * 4) + 4 + 2 + 1,),  # [2 teams * (N_PLAYERS players * 4 states)] + (ball * 4 states) + (score * 2) + time
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.game_state = GameState.create_default()
        self.states = [self.game_state]
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray):
        """Execute one time step within the environment."""
        team_a_actions, team_b_actions = self._action_to_team_actions(action)
        self.game_state = self.game_state.update(team_a_actions, team_b_actions)
        self.states.append(self.game_state)
        observation = self._get_observation()
        return observation, self._get_reward(), self._done, False, {}
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self.visualizer.get_rgb_array(self.soccer_state.state)
    
    def _get_reward(self) -> float:
        """Calculate reward for the current state."""
        reward = 0.0
        if self.sparse_rewards:
            previous_score = self.states[-2].score
            current_score = self.game_state.score
            if current_score[0] > previous_score[0]:
                reward += 1.0  # Team A scored
            if current_score[1] > previous_score[1]:
                reward -= 1.0  # Team B scored (bad for A)
        else:
            reward += self.game_state.score[0] - self.game_state.score[1]

        if self.sparse_rewards:
            previous_closeness_score = self.states[-2].team_b_avg_distance_to_ball - self.states[-2].team_a_avg_distance_to_ball
            current_closeness_score = self.game_state.team_b_avg_distance_to_ball - self.game_state.team_a_avg_distance_to_ball
            closeness_reward = current_closeness_score - previous_closeness_score
            reward += CLOSENESS_REWARD_MULTIPLIER * closeness_reward
        else:
            reward += CLOSENESS_REWARD_MULTIPLIER * (self.game_state.team_b_avg_distance_to_ball - self.game_state.team_a_avg_distance_to_ball)
        return reward
    
    @property
    def _done(self) -> bool:
        """Check if the game is done."""
        return self.game_state.time_remaining <= 0
            
    def _action_to_team_actions(self, action: np.ndarray) -> Tuple[List[PlayerAction], List[PlayerAction]]:
        """Reshape action into team actions."""
        action = action.reshape(N_PLAYERS * 2, 4)
        all_actions = [
            PlayerAction(
                vx=action[i][0] * MAX_VELOCITY,
                vy=action[i][1] * MAX_VELOCITY,
                kx=action[i][2] * MAX_KICK_FORCE,
                ky=action[i][3] * MAX_KICK_FORCE
            )
            for i in range(N_PLAYERS * 2)
        ]
        return all_actions[:N_PLAYERS], all_actions[N_PLAYERS:]
    
    def _get_observation(self) -> np.ndarray:
        """Convert game state to observation array."""
        state = self.game_state
        obs = []    
        for player in state.team_a:
            obs.extend([
                player.position[0] / FIELD_WIDTH,
                player.position[1] / FIELD_HEIGHT,
                player.velocity[0] / MAX_VELOCITY,
                player.velocity[1] / MAX_VELOCITY
            ])  
        for player in state.team_b:
            obs.extend([
                player.position[0] / FIELD_WIDTH,
                player.position[1] / FIELD_HEIGHT,
                player.velocity[0] / MAX_VELOCITY,
                player.velocity[1] / MAX_VELOCITY
            ])
        obs.extend([
            state.ball.position[0] / FIELD_WIDTH,
            state.ball.position[1] / FIELD_HEIGHT,
            state.ball.velocity[0] / MAX_VELOCITY,
            state.ball.velocity[1] / MAX_VELOCITY
        ])
        obs.extend([
            state.score[0],
            state.score[1]
        ])
        obs.append(state.time_remaining / GAME_DURATION)
        return np.array(obs, dtype=np.float32)

