# Game settings
N_PLAYERS = 5
MIN_KICKING_DISTANCE = 3
GAME_DURATION = 90  # seconds
GAME_LENGTH = GAME_DURATION

# Physics settings
MAX_VELOCITY = 2.0
MAX_KICK_FORCE = 4.0
BALL_FRICTION = 0.9
PLAYER_RADIUS = 2.0
BALL_RADIUS = 1.0

# Field dimensions
FIELD_WIDTH = 100
FIELD_HEIGHT = 60

# Goal settings
GOAL_WIDTH = 2
GOAL_HEIGHT = 20
GOAL_Y_MIN = 20
GOAL_Y_MAX = 40

# Team A starting positions (left side)
TEAM_A_START_POSITIONS = [
    (20, 30),
    (20, 20),
    (20, 40),
    (30, 25),
    (30, 35)
]

# Team B starting positions (right side)
TEAM_B_START_POSITIONS = [
    (80, 30),
    (80, 20),
    (80, 40),
    (70, 25),
    (70, 35)
]

# Ball starting position (center)
BALL_START_POSITION = (FIELD_WIDTH / 2, FIELD_HEIGHT / 2)

# Visualization settings
TEAM_A_COLOR = 'red'
TEAM_B_COLOR = 'blue'
BALL_COLOR = 'white'
FIELD_COLOR = '#2e8b57'
FIELD_ALPHA = 0.8
LINE_COLOR = 'white'

# Field markings
PENALTY_WIDTH = 20
PENALTY_HEIGHT = 40
CENTER_CIRCLE_RADIUS = 10
SPOT_RADIUS = 0.5

# Figure settings
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 7
FIGURE_DPI = 100 