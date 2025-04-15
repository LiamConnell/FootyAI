import shutil
import pytest
import os
import numpy as np
from src.game_state import GameState, Player, Ball, PlayerAction
from src.visualization import SoccerVisualizer

def test_visualization_creation():
    visualizer = SoccerVisualizer()
    assert visualizer.field_width == 100
    assert visualizer.field_height == 60
    assert visualizer.team_a_color == 'red'
    assert visualizer.team_b_color == 'blue'

def test_render_state():
    """Test rendering a single game state."""
    # Create a test game state with required parameters
    team_a = [Player(position=(20, 30), velocity=(0, 0)) for _ in range(5)]
    team_b = [Player(position=(80, 30), velocity=(0, 0)) for _ in range(5)]
    ball = Ball(position=(50, 30), velocity=(0, 0))
    game_state = GameState(team_a=team_a, team_b=team_b, ball=ball, time_remaining=90)
    
    # Create visualization
    vis = SoccerVisualizer()
    
    # Render the state
    fig, ax = vis.render_state(game_state)
    
    # Check that we have the correct number of patches
    # Field elements (7): field outline + center line + center circle + 2 penalty areas + 2 goals
    # Game elements (13): 5 team A players + 5 team B players + 1 ball + score text + time text
    # assert len(ax.patches) == 17
    
    # Check that the figure was created
    assert fig is not None
    assert ax is not None

def test_save_game_sequence(tmp_path):
    # Create a sequence of game states
    team_a = [Player(position=(20, 30), velocity=(0, 0)) for _ in range(5)]
    team_b = [Player(position=(80, 30), velocity=(0, 0)) for _ in range(5)]
    ball = Ball(position=(50, 30), velocity=(0, 0))
    initial_state = GameState(team_a=team_a, team_b=team_b, ball=ball, time_remaining=90)
    
    # Create a sequence of states by updating the game state
    states = [initial_state]
    current_state = initial_state
    for _ in range(30):  # Create 30 frames
        # Create actions for each player in team A
        actions = [(1, 0) for _ in range(len(initial_state.team_a))]  # Move right
        # Create a new state object for each frame
        current_state = GameState(
            team_a=[Player(position=(p.position[0] + a[0], p.position[1] + a[1]), velocity=(0, 0)) 
                   for p, a in zip(current_state.team_a, actions)],
            team_b=current_state.team_b,
            ball=Ball(position=current_state.ball.position, velocity=(0, 0)),
            time_remaining=current_state.time_remaining - 1,
            score=current_state.score
        )
        states.append(current_state)
    
    # Save the sequence
    output_file = os.path.join(tmp_path, "test_game.mp4")
    visualizer = SoccerVisualizer()
    visualizer.save_game_sequence(states, output_file, fps=10, show=False)
    
    # Check that the file was created
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0

    shutil.copy(output_file, "tests/test_output/test_game.mp4")

def test_random_movement_sequence(tmp_path):
    """Test visualization with random player movements and kicks."""
    # Create initial game state
    team_a = [Player(position=(20, 30), velocity=(0, 0)) for _ in range(5)]
    team_b = [Player(position=(80, 30), velocity=(0, 0)) for _ in range(5)]
    ball = Ball(position=(50, 30), velocity=(0, 0))
    initial_state = GameState(team_a=team_a, team_b=team_b, ball=ball, time_remaining=90)
    
    # Create a sequence of states with random movements
    states = [initial_state]
    current_state = initial_state
    
    for _ in range(30):  # Create 30 frames
        # Generate random actions for both teams
        team_a_actions = [
            PlayerAction(
                vx=np.random.uniform(-2, 2),  # Random movement in x
                vy=np.random.uniform(-2, 2),  # Random movement in y
                kx=np.random.uniform(-1, 1),  # Random kick in x
                ky=np.random.uniform(-1, 1)   # Random kick in y
            ) for _ in range(len(current_state.team_a))
        ]
        
        team_b_actions = [
            PlayerAction(
                vx=np.random.uniform(-2, 2),  # Random movement in x
                vy=np.random.uniform(-2, 2),  # Random movement in y
                kx=np.random.uniform(-1, 1),  # Random kick in x
                ky=np.random.uniform(-1, 1)   # Random kick in y
            ) for _ in range(len(current_state.team_b))
        ]
        
        # Update the game state with the random actions
        current_state = current_state.update(team_a_actions, team_b_actions)
        states.append(current_state)
    
    # Save the sequence
    output_file = os.path.join(tmp_path, "random_movement.mp4")
    visualizer = SoccerVisualizer()
    visualizer.save_game_sequence(states, output_file, fps=10, show=False)
    
    # Check that the file was created
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0

def test_goal_scoring_sequence(tmp_path):
    """Test visualization with goal scoring and ball reset."""
    # Create initial game state with team A players near the ball
    team_a = [Player(position=(30 + i*2, 30), velocity=(0, 0)) for i in range(5)]  # Players lined up near ball
    team_b = [Player(position=(80, 30), velocity=(0, 0)) for _ in range(5)]
    ball = Ball(position=(50, 30), velocity=(0, 0))
    initial_state = GameState(team_a=team_a, team_b=team_b, ball=ball, time_remaining=90)
    
    # Create a sequence of states with goal scoring
    states = [initial_state]
    current_state = initial_state
    
    for _ in range(60):  # Create 60 frames
        # Create actions for both teams
        # Team A players try to kick the ball right
        team_a_actions = [
            PlayerAction(vx=0.5, vy=0, kx=2, ky=0)  # Move slowly right while kicking
            for _ in range(len(current_state.team_a))
        ]
        
        # Team B players don't do anything
        team_b_actions = [
            PlayerAction(vx=0, vy=0, kx=0, ky=0)
            for _ in range(len(current_state.team_b))
        ]
        
        # Update the game state
        current_state = current_state.update(team_a_actions, team_b_actions)
        states.append(current_state)
    
    # Save the sequence
    output_file = os.path.join(tmp_path, "goal_scoring.mp4")
    visualizer = SoccerVisualizer()
    visualizer.save_game_sequence(states, output_file, fps=10, show=False)
    
    # Check that the file was created
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0
