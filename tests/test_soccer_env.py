import pytest
import numpy as np
import gymnasium as gym
from src.envs.soccer_env import SoccerEnv

def test_env_creation():
    """Test basic environment creation and properties."""
    env = SoccerEnv()
    
    # Test action space
    assert env.action_space.shape == (20,)  # 5 players * 4 actions
    assert np.all(env.action_space.low == np.array([-2, -2, -1, -1] * 5))
    assert np.all(env.action_space.high == np.array([2, 2, 1, 1] * 5))
    
    # Test observation space
    assert env.observation_space.shape == (44,)  # 11 entities * 4 values
    assert np.all(env.observation_space.low == np.array([0, 0, -2, -2] * 11))
    assert np.all(env.observation_space.high == np.array([100, 60, 2, 2] * 11))

def test_reset():
    """Test environment reset functionality."""
    env = SoccerEnv()
    obs, info = env.reset()
    
    # Test observation shape
    assert obs.shape == (44,)
    
    # Test initial state info
    assert info["score"] == (0, 0)
    assert info["time_remaining"] == 90
    
    # Test initial positions
    # Team A should be on the left
    for i in range(0, 20, 4):  # Check x positions of team A
        assert obs[i] < 50  # x position should be less than center
    
    # Team B should be on the right
    for i in range(20, 40, 4):  # Check x positions of team B
        assert obs[i] > 50  # x position should be greater than center
    
    # Ball should be in center
    assert obs[40] == 50  # Ball x position
    assert obs[41] == 30  # Ball y position

def test_step():
    """Test environment step functionality."""
    env = SoccerEnv()
    obs, info = env.reset()
    
    # Create a simple action: all players move right and kick right
    # vx=1, vy=0 (movement within [-2,2])
    # kx=0.5, ky=0 (kick within [-1,1])
    action = np.array([1.0, 0.0, 0.5, 0.0] * 5, dtype=np.float32)  # Only 5 players * 4 actions
    
    # Take a step and verify the output
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Test observation shape and basic properties
    assert obs.shape == (44,)  # 5 players * 4 values + 5 opponents * 4 values + ball position/velocity
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert 'score' in info
    assert 'time_remaining' in info

def test_goal_scoring():
    """Test goal scoring mechanics."""
    env = SoccerEnv()
    obs, info = env.reset()
    
    # Create an action that will score a goal
    # Move all players right (vx=2) and kick right (kx=1)
    action = np.array([2.0, 0.0, 1.0, 0.0] * 5, dtype=np.float32)  # Maximum allowed movement and kick
    
    # Run until goal is scored or time runs out
    for _ in range(90):
        obs, reward, terminated, truncated, info = env.step(action)
        if info["score"] != (0, 0):
            break
    
    # Test that a goal was scored
    assert info["score"] != (0, 0)
    
    # Test ball reset to center
    assert obs[40] == 50  # Ball x position
    assert obs[41] == 30  # Ball y position

def test_render():
    """Test rendering functionality."""
    env = SoccerEnv()
    env.reset()
    
    # Test that close doesn't raise an error
    env.close()

def test_invalid_actions():
    """Test handling of invalid actions."""
    env = SoccerEnv()
    env.reset()
    
    # Test action too large
    action = np.array([3, 3, 2, 2] * 5)  # Values outside action space
    with pytest.raises(AssertionError):
        env.step(action)
    
    # Test action too small
    action = np.array([-3, -3, -2, -2] * 5)  # Values outside action space
    with pytest.raises(AssertionError):
        env.step(action)

def test_game_termination():
    """Test game termination after time runs out."""
    env = SoccerEnv()
    obs, info = env.reset()
    
    # Run until game ends
    for _ in range(90):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    
    # Test that game terminated
    assert terminated
    assert info["time_remaining"] <= 0 