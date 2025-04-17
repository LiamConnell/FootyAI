import random
import uuid
from pydantic import BaseModel, Field
from typing import List, Tuple
import math

from src.config import (
    GOAL_HEIGHT,
    N_PLAYERS,
    MIN_KICKING_DISTANCE,
    GAME_DURATION,
    FIELD_WIDTH,
    FIELD_HEIGHT,
    TEAM_A_START_POSITIONS,
    TEAM_B_START_POSITIONS,
    BALL_START_POSITION
)

class PlayerAction(BaseModel):
    """Represents a player's action in a single frame."""
    vx: float = Field(..., description="Velocity in x direction")
    vy: float = Field(..., description="Velocity in y direction")
    kx: float = Field(..., description="Kick force in x direction")
    ky: float = Field(..., description="Kick force in y direction")

class Player(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    position: Tuple[float, float] = Field(..., description="Player's position on the field (x, y)")
    velocity: Tuple[float, float] = Field(..., description="Player's velocity vector (vx, vy)")

class Ball(BaseModel):
    position: Tuple[float, float] = Field(..., description="Ball's position on the field (x, y)")
    velocity: Tuple[float, float] = Field(..., description="Ball's velocity vector (vx, vy)")

class GameState(BaseModel):
    team_a: List[Player] = Field(..., description="List of players in team A")
    team_b: List[Player] = Field(..., description="List of players in team B")
    ball: Ball = Field(..., description="The ball in play")
    score: Tuple[int, int] = Field((0, 0), description="Score of the game (team_a, team_b)")
    time_remaining: float = Field(..., description="Time remaining in the game in seconds")

    @classmethod
    def create_default(cls, random_positions=True) -> 'GameState':
        """Create a default game state with players and ball in starting positions."""
        # Create team A players (left side)
        if random_positions:
            team_a = [
                Player(position=(random.random() * FIELD_WIDTH, random.random() * FIELD_HEIGHT), velocity=(0, 0))
                for _ in range(N_PLAYERS)
            ] 
            team_b = [
                Player(position=(random.random() * FIELD_WIDTH, random.random() * FIELD_HEIGHT), velocity=(0, 0))
                for _ in range(N_PLAYERS)
            ]
        else:
            team_a = [
                Player(position=pos, velocity=(0, 0))
                for pos in TEAM_A_START_POSITIONS
            ]
            
            # Create team B players (right side)
            team_b = [
                Player(position=pos, velocity=(0, 0))
                for pos in TEAM_B_START_POSITIONS
            ]
        
        # Create ball in center
        ball = Ball(position=BALL_START_POSITION, velocity=(0, 0))
        
        return cls(
            team_a=team_a,
            team_b=team_b,
            ball=ball,
            score=(0, 0),
            time_remaining=GAME_DURATION
        )

    def _distance_to_ball(self, player_pos: Tuple[float, float]) -> float:
        """Calculate the Euclidean distance between a player and the ball."""
        dx = player_pos[0] - self.ball.position[0]
        dy = player_pos[1] - self.ball.position[1]
        return math.sqrt(dx * dx + dy * dy)

    def update(self, team_a_actions: List[PlayerAction], team_b_actions: List[PlayerAction]) -> 'GameState':
        """
        Update the game state based on player actions.
        
        Args:
            team_a_actions: List of actions for team A players
            team_b_actions: List of actions for team B players
            
        Returns:
            A new GameState with updated positions and velocities
        """
        # Update player positions and velocities
        new_team_a = []
        for p, a in zip(self.team_a, team_a_actions):
            # Calculate new position
            new_x = p.position[0] + a.vx
            new_y = p.position[1] + a.vy
            
            # Clamp to field boundaries
            new_x = max(0, min(new_x, FIELD_WIDTH))
            new_y = max(0, min(new_y, FIELD_HEIGHT))
            
            # Only update velocity if not at boundary
            new_vx = a.vx if 0 < new_x < FIELD_WIDTH else 0
            new_vy = a.vy if 0 < new_y < FIELD_HEIGHT else 0
            
            new_team_a.append(Player(
                position=(new_x, new_y),
                velocity=(new_vx, new_vy)
            ))
        
        new_team_b = []
        for p, a in zip(self.team_b, team_b_actions):
            # Calculate new position
            new_x = p.position[0] + a.vx
            new_y = p.position[1] + a.vy
            
            # Clamp to field boundaries
            new_x = max(0, min(new_x, FIELD_WIDTH))
            new_y = max(0, min(new_y, FIELD_HEIGHT))
            
            # Only update velocity if not at boundary
            new_vx = a.vx if 0 < new_x < FIELD_WIDTH else 0
            new_vy = a.vy if 0 < new_y < FIELD_HEIGHT else 0
            
            new_team_b.append(Player(
                position=(new_x, new_y),
                velocity=(new_vx, new_vy)
            ))
        
        # Update ball position based on kicks from players within range
        ball_x, ball_y = self.ball.position
        ball_vx, ball_vy = self.ball.velocity
        
        # Sum up all kick forces from players within kicking distance
        total_kx = sum(
            a.kx for p, a in zip(self.team_a, team_a_actions)
            if self._distance_to_ball(p.position) <= MIN_KICKING_DISTANCE
        ) + sum(
            a.kx for p, a in zip(self.team_b, team_b_actions)
            if self._distance_to_ball(p.position) <= MIN_KICKING_DISTANCE
        )
        
        total_ky = sum(
            a.ky for p, a in zip(self.team_a, team_a_actions)
            if self._distance_to_ball(p.position) <= MIN_KICKING_DISTANCE
        ) + sum(
            a.ky for p, a in zip(self.team_b, team_b_actions)
            if self._distance_to_ball(p.position) <= MIN_KICKING_DISTANCE
        )
        
        # Update ball velocity and position
        new_ball_vx = ball_vx + total_kx
        new_ball_vy = ball_vy + total_ky
        
        # Calculate new ball position
        new_ball_x = ball_x + new_ball_vx
        new_ball_y = ball_y + new_ball_vy

        # Check for goals
        new_score = list(self.score)
        ball_within_y_goal_range = (
            (new_ball_y >= (FIELD_HEIGHT - GOAL_HEIGHT) / 2) and 
            (new_ball_y <= (FIELD_HEIGHT + GOAL_HEIGHT) / 2)
        )
        if ball_within_y_goal_range and new_ball_x <= 0:  # Team B scores
            new_score[1] += 1
            new_ball_x = FIELD_WIDTH / 2  # Reset to center
            new_ball_y = FIELD_HEIGHT / 2
            new_ball_vx = 0
            new_ball_vy = 0
        elif ball_within_y_goal_range and new_ball_x >= FIELD_WIDTH:  # Team A scores
            new_score[0] += 1
            new_ball_x = FIELD_WIDTH / 2  # Reset to center
            new_ball_y = FIELD_HEIGHT / 2
            new_ball_vx = 0
            new_ball_vy = 0
        
        # Handle ball bouncing off walls
        if new_ball_x <= 0 or new_ball_x >= FIELD_WIDTH:
            new_ball_vx = -new_ball_vx * 0.8  # Bounce with some energy loss
            new_ball_x = max(0, min(new_ball_x, FIELD_WIDTH))
        
        if new_ball_y <= 0 or new_ball_y >= FIELD_HEIGHT:
            new_ball_vy = -new_ball_vy * 0.8  # Bounce with some energy loss
            new_ball_y = max(0, min(new_ball_y, FIELD_HEIGHT))
        
        return GameState(
            team_a=new_team_a,
            team_b=new_team_b,
            ball=Ball(
                position=(new_ball_x, new_ball_y),
                velocity=(new_ball_vx, new_ball_vy)
            ),
            score=tuple(new_score),
            time_remaining=self.time_remaining - 1
        )

    @property
    def team_a_avg_distance_to_ball(self) -> float:
        return sum(self._distance_to_ball(p.position) for p in self.team_a) / len(self.team_a)

    @property
    def team_b_avg_distance_to_ball(self) -> float:
        return sum(self._distance_to_ball(p.position) for p in self.team_b) / len(self.team_b)
