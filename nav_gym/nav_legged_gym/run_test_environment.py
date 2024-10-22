from nav_gym.nav_legged_gym.envs.legged_env import LeggedEnv
from nav_gym.nav_legged_gym.envs.test_env import TestEnv
from nav_gym.nav_legged_gym.envs.legged_env_config import LeggedEnvCfg
import torch

if __name__ == "__main__":
    env = TestEnv(LeggedEnvCfg())
    env.reset()
    while True:
        env.step(torch.zeros((env.num_envs, env.num_actions)).to(env.device))
        env.render()

