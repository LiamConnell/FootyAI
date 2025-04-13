from src.envs.soccer_env import SoccerEnv

def main():
    env = SoccerEnv(render_mode="human")
    env.reset()
    env.render()
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
    
    env.close()
    
