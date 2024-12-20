import os
import time
import pygame

# Your existing imports
# from legged_gym import LeggedNavEnv, LeggedNavEnvCfg, OnPolicyRunner, TrainConfig, class_to_dict
from nav_gym.nav_legged_gym.envs.locomotion_pae_latent_scan_residual_command_env import LocomotionPAELatentScanEnv
from nav_gym.nav_legged_gym.envs.config_local_nav_pae_latent_scan_residual_command_env import LocalNavPAEEnvCfg
from nav_gym.nav_legged_gym.envs.local_nav_pae_latent_scan_residual_command_env import LocalNavPAEEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_local_nav_pae_latent_scan_command import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.nav_legged_gym.test.interactive_module import InteractModuleVelocity,InteractModulePosition
from nav_gym import NAV_GYM_ROOT_DIR
import nav_gym.nav_legged_gym.common.rewards.rewards as R
import torch
import os
import time
if __name__ == "__main__":
    interact_module = InteractModulePosition()

    # Set up your environment and policy
    log_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/" + time.strftime("%Y%m%d-%H%M%S"))
    checkpoint_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/local_nav/pae_latent_scan_residual_command/cluster_1220_1/" + "model_600.pt")
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)

    env_cfg = LocalNavPAEEnvCfg()
    env_cfg.env.num_envs = 1

    env = LocalNavPAEEnv(env_cfg, LocomotionPAELatentScanEnv)
    env.set_flag_enable_resample_pos(False)
    env.set_flag_enable_resample_vel(True)
    env.set_flag_enable_reset(False)
    env.ll_env.set_flag_enable_resample(False)
    env.ll_env.set_flag_enable_reset(False)
    runner = OnPolicyRunner(env,train_cfg_dict , log_dir=log_dir, device="cuda:0")
    runner.load(checkpoint_dir)
    policy = runner.get_inference_policy()
    obs, extras = env.reset()

    # Main loop
    running = True
    while running:
        interact_module.update()
        # Update the commands in the environment
        # Assuming 'set_velocity_commands' correctly sets the commands for all environments
        # print("[INFO][test_local_nav_interact]Setting velocity commands")
        env.set_goal_position_command(interact_module.x_goal,interact_module.y_goal,env.robot.root_pos_w[0,2])

        # Print the current commands
        print(f"Current Commands - X: {interact_module.x_goal}, Y: {interact_module.y_goal}, Z: {interact_module.z_goal}")

        # Run the policy and step the environment
        action = policy(obs)
        obs, _, _, extras = env.step(action)
        env.ll_env.render()

        # Update the Pygame display (not strictly necessary unless you're drawing something)
        pygame.display.flip()

    # Clean up
    pygame.quit()
    