from gymnasium.envs.registration import register

register(
    id="Soccer-v0",
    entry_point="src.envs.soccer_env:SoccerEnv",
    max_episode_steps=90,  # 90 seconds of game time
) 