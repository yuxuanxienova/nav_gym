from nav_gym.nav_legged_gym.envs.config_locomotion_env import LocomotionEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
import torch

if __name__ == "__main__":
    env = LocomotionEnv(LocomotionEnvCfg())
    env.reset()
    while True:
        env.step(torch.zeros((env.num_envs, env.num_actions)).to(env.device))
        env.render()

