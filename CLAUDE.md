# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
FootyAI is a multi-agent adversarial reinforcement learning project that simulates competitive soccer gameplay. Two teams of agents learn through self-play using policy gradient methods, with continuous state and action spaces.

## Architecture

### Core Components
- **Environment**: `src/v2/torch_soccer_env.py` - Vectorized PyTorch implementation of the soccer simulation
- **Policy Network**: `src/policy_network.py` - Neural network that outputs actions for agents
- **Training**: `src/v2/train.py` - Main training loop using policy gradients with self-play
- **Configuration**: `src/config.py` - Game settings, physics parameters, and field dimensions
- **Visualization**: `src/visualization.py` - Renders game states to MP4 videos

### Directory Structure
- `src/v1/` - Earlier single-environment implementation
- `src/v2/` - Current vectorized batch training implementation
- `src/_old/` - Legacy code and experiments
- `tests/` - Unit tests for core components
- `docs/` - MkDocs documentation site

## Development Commands

### Testing
```bash
pytest                    # Run all tests
pytest tests/test_*.py    # Run specific test file
pytest -v                 # Verbose output
pytest --cov=src          # Run with coverage
```

### Code Quality
```bash
black src/                # Format code
isort src/                # Sort imports
mypy src/                 # Type checking
ruff src/                 # Linting
```

### Training
```bash
python -m src.v2.train    # Run local training
uv run python deploy_vertex_ai.py  # Deploy to Vertex AI
```

### Vertex AI Deployment
```bash
# Deploy and monitor
uv run python deploy_vertex_ai.py
gcloud ai custom-jobs describe <JOB_ID> --format="value(state)"  # Check status
gcloud logging read 'resource.labels.job_id="<JOB_ID>"' --project learnagentspace --limit 100  # View logs
gsutil -m cp -r gs://footyai/videos/v2_torch_soccer_<TIMESTAMP>/ videos/  # Download videos
```

### Documentation
```bash
mkdocs serve              # Serve docs locally
mkdocs build              # Build static site
```

## Key Implementation Details

### Environment States
- Observation space: Player positions, velocities, ball position/velocity, game state
- Action space: 4D per player (x/y movement, x/y kick force)
- Rewards: Goal scoring (+1/-1), optional ball proximity and possession bonuses

### Training Loop
- Uses vectorized environments (batch_size=512 by default)
- Policy gradient with discounted rewards
- Self-play between identical networks with team one-hot encoding
- Periodic video rendering for monitoring progress

### Configuration
Key parameters in `src/config.py`:
- `N_PLAYERS = 5` players per team
- `GAME_DURATION = 200` time steps per game
- Field dimensions, physics constants, starting positions

## Dependencies
Core dependencies managed in `requirements.txt`:
- PyTorch for neural networks and vectorized computation
- Gymnasium for RL environment interface
- Pydantic for data validation
- MkDocs for documentation

## Testing Strategy
Tests cover:
- Game state transitions and physics
- Environment reset and step functions
- Visualization rendering
- Core game logic and rules

## Experiment Issues

Issues labeled `experiment` document training runs and experimental configurations. They should include:
- Job ID and monitoring commands
- Complete deployment configuration (epochs, batch size, machine type)  
- Git commit hash for reproducibility
- Expected outcome and current status