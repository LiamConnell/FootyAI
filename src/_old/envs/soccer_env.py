from datetime import datetime
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional

from src.game_state import GameState, Player, Ball, PlayerAction

class SoccerEnv(gym.Env):
    """A soccer environment for reinforcement learning."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Action space: for each player, [vx, vy, kx, ky]
        # vx, vy: movement velocity (-2 to 2)
        # kx, ky: kick force (-1 to 1)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1] * 10),  # 5 players per team
            high=np.array([1, 1, 1, 1] * 10),
            dtype=np.float32
        )
        
        # Observation space: positions and velocities of all players and ball
        # [x, y, vx, vy] for each player (5 per team) and ball
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -2, -2] * 11),  # 10 players + 1 ball
            high=np.array([100, 60, 2, 2] * 11),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.visualizer = None
        if render_mode in ["human", "rgb_array"]:
            from src.visualization import SoccerVisualizer
            self.visualizer = SoccerVisualizer()
        
        # Initialize game state
        self.reset()
    
    def _get_obs(self) -> np.ndarray:
        """Convert game state to observation array."""
        obs = []
        
        # Add team A players
        for player in self.state.team_a:
            obs.extend([player.position[0], player.position[1], 
                       player.velocity[0], player.velocity[1]])
        
        # Add team B players
        for player in self.state.team_b:
            obs.extend([player.position[0], player.position[1], 
                       player.velocity[0], player.velocity[1]])
        
        # Add ball
        obs.extend([self.state.ball.position[0], self.state.ball.position[1],
                   self.state.ball.velocity[0], self.state.ball.velocity[1]])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information about the game state."""
        return {
            "score": self.state.score,
            "time_remaining": self.state.time_remaining
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Create initial game state
        team_a = [Player(position=(20, 30), velocity=(0, 0)) for _ in range(5)]
        team_b = [Player(position=(80, 30), velocity=(0, 0)) for _ in range(5)]
        ball = Ball(position=(50, 30), velocity=(0, 0))
        self.state = GameState(team_a=team_a, team_b=team_b, ball=ball, time_remaining=90)

        self.states = [self.state]
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        # Validate action
        if not self.action_space.contains(action):
            raise AssertionError(f"Action {action} is not in action space {self.action_space}")
        
        # Reshape action into team A and team B actions
        team_a_actions = []
        team_b_actions = []

        run_multiplier = 5
        kick_multiplier = 10
        
        for i in range(5):  # 5 players per team
            # Team A actions
            team_a_actions.append(PlayerAction(
                vx=action[i*4] * run_multiplier,
                vy=action[i*4 + 1] * run_multiplier,
                kx=action[i*4 + 2] * kick_multiplier,
                ky=action[i*4 + 3] * kick_multiplier
            ))
            
            # Team B actions (using same action for now - could be modified for different strategies)
            team_b_actions.append(PlayerAction(
                vx=action[i*4 + 4] * run_multiplier,
                vy=action[i*4 + 5] * run_multiplier,
                kx=action[i*4 + 6] * kick_multiplier,
                ky=action[i*4 + 7] * kick_multiplier
            ))
        
        # Update game state
        self.state = self.state.update(team_a_actions, team_b_actions)
        self.states.append(self.state)
        
        # Calculate reward (simple reward for now - could be enhanced)
        reward = float(self.state.score[0] - self.state.score[1])  # Team A's score minus Team B's score
        reward += 1e-3 * (self.state.team_b_avg_distance_to_ball - self.state.team_a_avg_distance_to_ball)

        # Check if game is done
        done = self.state.time_remaining <= 0
        
        return self._get_obs(), reward, done, False, self._get_info()
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        # if self.render_mode is None:
        #     return
        
        # if self.render_mode == "human":
        #     self.visualizer.render_state(self.state)
        #     return None
        # elif self.render_mode == "rgb_array":
        #     # Return RGB array for video recording
        #     rgb_array =  self.visualizer.get_rgb_array(self.state)
            
        #     return rgb_array
        if self.state.time_remaining == 1: 
            print("Saving game sequence")
            self.visualizer.save_game_sequence(self.states, f"videos/soccer_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            return 1,2
    
    def close(self):
        """Close the environment."""
        # if self.visualizer is not None:
        #     self.visualizer.close() 
        pass