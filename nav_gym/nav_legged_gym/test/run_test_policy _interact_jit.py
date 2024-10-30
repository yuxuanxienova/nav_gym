import os
import time
import pygame

# Your existing imports
# from legged_gym import LeggedNavEnv, LeggedNavEnvCfg, OnPolicyRunner, TrainConfig, class_to_dict
from nav_gym.nav_legged_gym.envs.legged_nav_env_config import LeggedNavEnvCfg
from nav_gym.nav_legged_gym.envs.legged_nav_env import LeggedNavEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.train_config import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
import torch
import os
import time
from nav_gym import NAV_GYM_ROOT_DIR
if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    # Create a small window to capture keyboard inputs
    screen = pygame.display.set_mode((200, 200))
    pygame.display.set_caption("Robot Controller")

    # Set up your environment and policy
    scripted_model_path = os.path.join(NAV_GYM_ROOT_DIR, "resources/model/low_level/" )
    file_name = "ll_jit_model.pt"
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)

    env = LeggedNavEnv(LeggedNavEnvCfg())
    env.play_mode = True

    #Load the policy
    try:
        policy = torch.jit.load(scripted_model_path + file_name).to("cuda:0")
        policy.eval()
    except Exception as e:
        print("Loading scripted model failed:", e)
        exit(1)

    obs, extras = env.reset()

    # Initialize velocities
    x_vel = 0.0
    y_vel = 0.0
    yaw_vel = 0.0

    # Define maximum velocities
    max_x_vel = 2.0
    max_y_vel = 1.0
    max_yaw_vel = 1.0  # Adjust this value based on your environment's limits

    # Main loop
    running = True
    while running:
        # Handle Pygame events
        for event in pygame.event.get():
            # Allow exiting the program
            if event.type == pygame.QUIT:
                running = False
                break
            # Handle key press events
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    x_vel = max_x_vel
                elif event.key == pygame.K_s:
                    x_vel = -max_x_vel
                elif event.key == pygame.K_a:
                    y_vel = -max_y_vel
                elif event.key == pygame.K_d:
                    y_vel = max_y_vel
                elif event.key == pygame.K_q:
                    yaw_vel = max_yaw_vel
                elif event.key == pygame.K_e:
                    yaw_vel = -max_yaw_vel
            # Handle key release events
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_w, pygame.K_s):
                    x_vel = 0.0
                elif event.key in (pygame.K_a, pygame.K_d):
                    y_vel = 0.0
                elif event.key in (pygame.K_q, pygame.K_e):
                    yaw_vel = 0.0

        # Update the commands in the environment
        # Assuming 'set_velocity_commands' correctly sets the commands for all environments
        env.set_velocity_commands(x_vel, y_vel, yaw_vel)

        # Print the current commands
        print(f"Current Commands - X: {x_vel}, Y: {y_vel}, Yaw: {yaw_vel}")

        # Run the policy and step the environment
        action = policy(obs)
        obs, _, _, _, extras = env.step(action)
        env.render()

        # Update the Pygame display (not strictly necessary unless you're drawing something)
        pygame.display.flip()

    # Clean up
    pygame.quit()
    