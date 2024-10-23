from nav_gym.nav_legged_gym.envs.legged_nav_env_config import LeggedNavEnvCfg
from nav_gym.nav_legged_gym.envs.legged_nav_env import LeggedNavEnv
import torch

if __name__ == "__main__":
    env = LeggedNavEnv(LeggedNavEnvCfg())
    env.reset()
    while True:
        env.step(torch.zeros((env.num_envs, env.num_actions)).to(env.device))
        env.render()

