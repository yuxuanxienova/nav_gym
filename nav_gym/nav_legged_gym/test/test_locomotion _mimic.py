import os
import time
import pygame

# Your existing imports
# from legged_gym import LeggedNavEnv, LeggedNavEnvCfg, OnPolicyRunner, TrainConfig, class_to_dict
from nav_gym.nav_legged_gym.envs.config_locomotion_mimic_env import LocomotionMimicEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_mimic_env import LocomotionMimicEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion_mimic import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
import torch
import os
import time
if __name__ == "__main__":

    # Set up your environment and policy
    log_dir = os.path.join(os.path.dirname(__file__), "logs/" + time.strftime("%Y%m%d-%H%M%S"))
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "logs/locomotion_mimic/20241127-034615/" + "model_10500.pt")
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)

    env_cfg = LocomotionMimicEnvCfg()
    env_cfg.env.num_envs = 1
    env_cfg.robot.randomization.randomize_friction = False
    env_cfg.randomization.push_robots = False
    env_cfg.terrain_unity.terrain_file = "/terrain/Plane1.obj"
    env_cfg.terrain_unity.translation = [0.0, 0.0, -1.0]
    env_cfg.terrain_unity.env_origin_pattern = "point"
    env_cfg.gym.viewer.eye = (3.0, 3.0, 3.0)

    env = LocomotionMimicEnv(env_cfg)
    env.set_flag_enable_reset(False)
    env.set_flag_enable_resample(False)
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    runner.load(checkpoint_dir)
    policy = runner.get_inference_policy()
    obs, extras = env.reset()

    # Main loop
    running = True
    while running:

        # Run the policy and step the environment
        action = policy(obs)
        obs, _, _, extras = env.step(action)
        env.render()


    