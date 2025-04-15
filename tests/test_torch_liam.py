import torch
import numpy as np
from src.torch_soccer_env import TorchSoccerEnv
from src.visualization import states_to_mp4
from src.config import FIELD_HEIGHT, FIELD_WIDTH, GOAL_HEIGHT

def run_simulation(scenario_name, actions_generator, steps=300, n_players=5):
    """
    Run a specific simulation scenario with given actions generator function
    
    Args:
        scenario_name: Name of the scenario for the output file
        actions_generator: Function that takes step index and returns actions tensor
        steps: Number of steps to simulate
        n_players: Number of players per team
    """
    print(f"Running scenario: {scenario_name}")
    
    # Initialize environment
    env = TorchSoccerEnv(batch_size=1, n_players=n_players, device="cpu")
    obs = env.reset()
    
    # Keep track of all game states
    game_states = []
    
    # Run simulation
    for step in range(steps):
        # Generate actions based on scenario
        actions = actions_generator(step, env)
        
        # Step the environment
        obs, reward, done, _ = env.step(actions)
        
        # Convert to GameState and save
        game_state = env.to_pydantic(batch_idx=0)
        game_states.append(game_state)
        
        # Print progress
        if step % 50 == 0:
            print(f"  Step {step}/{steps}, Score: {game_state.score}")
            
        if done:
            print("  Game finished!")
            break
    
    # Generate video
    output_path = f"soccer_simulation_{scenario_name}.mp4"
    states_to_mp4(game_states, output_path)
    print(f"Video saved to {output_path}")
    
    return game_states

def scenario_random_movement(steps=300, n_players=5):
    """Random player movements and kicks"""
    def actions_generator(step, env):
        return torch.randn(1, n_players*2, 4) * 1.0
    
    return run_simulation("random_movement", actions_generator, steps, n_players)

def scenario_team_rush(steps=300, n_players=5):
    """Team 1 rushes towards team 2's goal"""
    def actions_generator(step, env):
        actions = torch.zeros(1, n_players*2, 4)
        
        # Team 1 moves right (towards opponent's goal)
        actions[0, :n_players, 0] = 2.0  # vx
        
        # Team 2 moves randomly
        actions[0, n_players:, :2] = torch.randn(n_players, 2) * 0.5
        
        # Kick actions for team 1 (towards opponent's goal)
        actions[0, :n_players, 2] = 3.0  # kx
        
        return actions
    
    return run_simulation("team_rush", actions_generator, steps, n_players)

def scenario_pass_and_score(steps=300, n_players=5):
    """Team 1 forms a pattern to pass ball and score"""
    def actions_generator(step, env):
        actions = torch.zeros(1, n_players*2, 4)
        ball_pos = env.ball_position[0]  # Get ball position
        
        # Phase 1: Initial setup - move players to strategic positions
        if step < 50:
            # Team 1 - Position players in formation
            for i in range(n_players):
                target_x = 30 + (i * 10)  # Spread across the field
                target_y = 20 + (i * 5)   # Diagonal formation
                
                # Get current player position
                player_pos = env.team1_positions[0, i]
                
                # Move towards target
                dx = target_x - player_pos[0]
                dy = target_y - player_pos[1]
                
                # Set movement velocity (normalized and scaled)
                dist = (dx**2 + dy**2)**0.5
                if dist > 0.1:
                    actions[0, i, 0] = dx / dist * 1.5  # vx
                    actions[0, i, 1] = dy / dist * 1.5  # vy
        
        # Phase 2: Pass the ball between teammates
        elif step < 150:
            # Find nearest player to ball
            team1_pos = env.team1_positions[0]
            dists = torch.norm(team1_pos - ball_pos.unsqueeze(0), dim=1)
            nearest_idx = torch.argmin(dists).item()
            
            # Target player for pass (rotating through team)
            target_idx = (nearest_idx + 1) % n_players
            
            # Nearest player kicks towards target player
            if dists[nearest_idx] < 5.0:  # If close enough to kick
                kick_dir = team1_pos[target_idx] - team1_pos[nearest_idx]
                kick_norm = torch.norm(kick_dir)
                if kick_norm > 0:
                    actions[0, nearest_idx, 2] = kick_dir[0] / kick_norm * 3.0  # kx
                    actions[0, nearest_idx, 3] = kick_dir[1] / kick_norm * 3.0  # ky
            
            # All players move toward their assigned positions
            for i in range(n_players):
                # Players move toward the ball if they're the next target
                if i == target_idx:
                    target_x = ball_pos[0] + 5  # Slightly ahead of ball
                    target_y = ball_pos[1]
                else:
                    # Others maintain formation
                    target_x = 30 + (i * 10) 
                    target_y = 20 + (i * 5)
                    
                player_pos = env.team1_positions[0, i]
                dx = target_x - player_pos[0]
                dy = target_y - player_pos[1]
                
                dist = (dx**2 + dy**2)**0.5
                if dist > 0.1:
                    actions[0, i, 0] = dx / dist * 1.0  # vx
                    actions[0, i, 1] = dy / dist * 1.0  # vy
        
        # Phase 3: Rush to goal and shoot
        else:
            team1_pos = env.team1_positions[0]
            dists = torch.norm(team1_pos - ball_pos.unsqueeze(0), dim=1)
            nearest_idx = torch.argmin(dists).item()
            
            # Nearest player rushes to goal with ball
            if dists[nearest_idx] < 5.0:
                # Rush toward goal
                actions[0, nearest_idx, 0] = 2.0  # vx
                
                # Kick toward goal if close enough to right side
                if team1_pos[nearest_idx, 0] > 70:
                    actions[0, nearest_idx, 2] = 5.0  # Strong kick toward goal
            
            # Other players move toward goal as support
            for i in range(n_players):
                if i != nearest_idx:
                    actions[0, i, 0] = 1.5  # Move toward goal
                    
                    # Spread out vertically
                    if i % 2 == 0:
                        actions[0, i, 1] = 0.5  # Move down slightly 
                    else:
                        actions[0, i, 1] = -0.5  # Move up slightly
        
        # Team 2 defends (simple AI)
        for i in range(n_players):
            player_pos = env.team2_positions[0, i]
            
            # Move toward the ball
            dx = ball_pos[0] - player_pos[0]
            dy = ball_pos[1] - player_pos[1]
            
            dist = (dx**2 + dy**2)**0.5
            if dist > 0.1:
                actions[0, n_players + i, 0] = dx / dist * 0.8  # vx
                actions[0, n_players + i, 1] = dy / dist * 0.8  # vy
            
            # Try to kick the ball away if close
            if dist < 3.0:
                actions[0, n_players + i, 2] = -3.0  # Kick away from goal
                
        return actions
    
    return run_simulation("pass_and_score", actions_generator, steps, n_players)

def scenario_goalkeeper_save(steps=300, n_players=5):
    """Team 2 goalkeeper tries to save shots from team 1"""
    def actions_generator(step, env):
        actions = torch.zeros(1, n_players*2, 4)
        ball_pos = env.ball_position[0]
        ball_vel = env.ball_velocity[0]
        
        # Set one player from team 2 as goalkeeper
        goalkeeper_idx = 0
        
        # Position the goalkeeper
        goalkeeper_pos = env.team2_positions[0, goalkeeper_idx]
        goal_center_y = FIELD_HEIGHT / 2
        
        # Goalkeeper stays near their goal line but moves vertically to track the ball
        target_x = FIELD_WIDTH - 5  # Near the goal line
        target_y = ball_pos[1]  # Track the ball vertically
        
        # Limit goalkeeper's vertical movement to goal area
        target_y = max(goal_center_y - GOAL_HEIGHT/1.5, min(goal_center_y + GOAL_HEIGHT/1.5, target_y))
        
        # Move goalkeeper towards target position
        dx = target_x - goalkeeper_pos[0]
        dy = target_y - goalkeeper_pos[1]
        
        dist = (dx**2 + dy**2)**0.5
        if dist > 0.1:
            actions[0, n_players + goalkeeper_idx, 0] = dx / dist * 1.5  # vx
            actions[0, n_players + goalkeeper_idx, 1] = dy / dist * 1.5  # vy
        
        # Goalkeeper tries to kick the ball away if it's close
        if torch.norm(goalkeeper_pos - ball_pos) < 5.0:
            # Kick away from goal (to the left)
            actions[0, n_players + goalkeeper_idx, 2] = -4.0  # kx
        
        # Team 1 tries to score
        if step % 60 < 30:  # Move into position
            for i in range(n_players):
                # Position players for attack
                target_x = 60 + (i * 5)
                target_y = 20 + (i * 5)
                
                player_pos = env.team1_positions[0, i]
                dx = target_x - player_pos[0]
                dy = target_y - player_pos[1]
                
                dist = (dx**2 + dy**2)**0.5
                if dist > 0.1:
                    actions[0, i, 0] = dx / dist * 1.0  # vx
                    actions[0, i, 1] = dy / dist * 1.0  # vy
        else:  # Shoot at goal
            # Find nearest player to ball
            team1_pos = env.team1_positions[0]
            dists = torch.norm(team1_pos - ball_pos.unsqueeze(0), dim=1)
            nearest_idx = torch.argmin(dists).item()
            
            if dists[nearest_idx] < 5.0:
                # Calculate angle to shoot toward goal (aim at different parts of the goal)
                goal_target_y = goal_center_y + 5 * np.sin(step / 10)  # Oscillate target
                
                # Direction to goal
                dx = FIELD_WIDTH - team1_pos[nearest_idx, 0]
                dy = goal_target_y - team1_pos[nearest_idx, 1]
                
                kick_dist = (dx**2 + dy**2)**0.5
                if kick_dist > 0.1:
                    actions[0, nearest_idx, 2] = dx / kick_dist * 4.0  # Strong kick toward goal
                    actions[0, nearest_idx, 3] = dy / kick_dist * 4.0
        
        # Rest of team 2 defends
        for i in range(1, n_players):  # Skip goalkeeper
            player_pos = env.team2_positions[0, i]
            
            # Move toward the ball
            dx = ball_pos[0] - player_pos[0]
            dy = ball_pos[1] - player_pos[1]
            
            dist = (dx**2 + dy**2)**0.5
            if dist > 0.1:
                actions[0, n_players + i, 0] = dx / dist * 0.7  # vx
                actions[0, n_players + i, 1] = dy / dist * 0.7  # vy
            
        return actions
    
    return run_simulation("goalkeeper_save", actions_generator, steps, n_players)

def scenario_ball_bounce(steps=200, n_players=5):
    """Test ball bounce after diagonal kick by a player"""
    def actions_generator(step, env):
        actions = torch.zeros(1, n_players*2, 4)
        player_idx = 0  # Team 1's first player
        ball_pos = env.ball_position[0]
        player_pos = env.team1_positions[0, player_idx]

        # Step 0–10: Move player toward ball
        if step < 1000:
            dx = ball_pos[0] - player_pos[0]
            dy = ball_pos[1] - player_pos[1]
            dist = torch.norm(torch.tensor([dx, dy]))
            if dist > 0.1:
                actions[0, player_idx, 0] = dx / dist * 2.0  # vx
                actions[0, player_idx, 1] = dy / dist * 2.0  # vy
        
        # Step 10–15: Kick the ball diagonally up-right with full force
        # elif step == 10:
        actions[0, player_idx, 2] = 5.0  # kx
        actions[0, player_idx, 3] = 5.0  # ky
        
        return actions

    return run_simulation("ball_bounce", actions_generator, steps, n_players)


def tests_main():
    print("Starting soccer simulations...")
    
    # Run all scenarios
    scenario_random_movement(steps=200)
    scenario_team_rush(steps=250)
    scenario_pass_and_score(steps=300)
    scenario_goalkeeper_save(steps=350)
    scenario_ball_bounce(steps=200)
    
    print("All simulations completed!")
