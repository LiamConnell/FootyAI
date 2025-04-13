import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from src.envs.soccer_env import SoccerEnv

def make_env(env_id, rank, seed=0, render_mode=None):
    """Create a soccer environment."""
    def _init():
        env = SoccerEnv(render_mode=render_mode)
        # Wrap with Monitor to track episode rewards and lengths
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    # Create directories for logs and models
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    # Create training environment
    env = DummyVecEnv([make_env("Soccer-v0", i) for i in range(1)])
    
    # Create evaluation environment with video recording
    eval_env = DummyVecEnv([make_env("Soccer-v0", 0, render_mode="human")])
    
    # Create model
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="soccer_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=True
    )
    
    # Train the model
    model.learn(
        total_timesteps=1000000,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save the final model
    model.save("soccer_model_final")
    
    # Close environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main() 