import pytest
from src.game_state import GameState, Player, Ball, PlayerAction

def test_game_state_initialization():
    # Create a game state with initial values
    team_a = [Player(position=(0, 0), velocity=(0, 0)) for _ in range(5)]
    team_b = [Player(position=(0, 0), velocity=(0, 0)) for _ in range(5)]
    ball = Ball(position=(50, 50), velocity=(0, 0))
    game_state = GameState(team_a=team_a, team_b=team_b, ball=ball, time_remaining=90)

    # Check that the game state is initialized correctly
    assert len(game_state.team_a) == 5
    assert len(game_state.team_b) == 5
    assert game_state.ball.position == (50, 50)
    assert game_state.score == (0, 0)
    assert game_state.time_remaining == 90

def test_update_game_state():
    # Create a game state with initial values
    team_a = [Player(position=(0, 0), velocity=(0, 0)) for _ in range(5)]
    team_b = [Player(position=(0, 0), velocity=(0, 0)) for _ in range(5)]
    ball = Ball(position=(50, 50), velocity=(0, 0))
    game_state = GameState(team_a=team_a, team_b=team_b, ball=ball, time_remaining=90)

    # Define actions for both teams
    team_a_actions = [
        PlayerAction(vx=1, vy=0, kx=0, ky=0)  # Move right by 1 unit
        for _ in range(5)
    ]
    team_b_actions = [
        PlayerAction(vx=0, vy=0, kx=0, ky=0)  # No movement
        for _ in range(5)
    ]

    # Update the game state
    updated_state = game_state.update(team_a_actions, team_b_actions)

    # Check that the game state is updated correctly
    assert updated_state.team_a[0].position == (1, 0)  # Player moved right
    assert updated_state.team_b[0].position == (0, 0)  # Player didn't move
    assert updated_state.ball.position == (50, 50)  # Ball position unchanged
    assert updated_state.time_remaining == 89  # Time decremented 