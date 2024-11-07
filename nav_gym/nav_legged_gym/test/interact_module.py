import pygame
import time
from typing import Tuple
from nav_gym.nav_legged_gym.envs.config_local_nav_env import LocalNavEnvCfg
class InteractModule:
    def __init__(self,env_cfg:LocalNavEnvCfg=None):
        # Initialize Pygame
        pygame.init()
        # Create a small window to capture keyboard inputs
        screen = pygame.display.set_mode((200, 200))
        pygame.display.set_caption("Robot Controller")
        eps=1e-5
        # Initialize velocities
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.yaw_vel = 0.0

        # Define maximum velocities
        self.max_x_vel = 2
        self.max_y_vel = 1.0
        self.max_yaw_rate = 1.0# Adjust this value based on your environment's limits

        if env_cfg is not None:
            self.max_x_vel = env_cfg.max_x_vel-eps
            self.max_y_vel = env_cfg.max_y_vel-eps
            self.max_yaw_rate = env_cfg.max_yaw_rate-eps


        self.vel_cmd_max: Tuple[float, float, float] = (self.max_x_vel, self.max_y_vel, self.max_yaw_rate)  # x, y, yaw
        self.vel_cmd_scale: Tuple[float, float, float] = (2.0 * self.max_x_vel, 2.0 * self.max_y_vel, 2.0 * self.max_yaw_rate)  # x, y, yaw
        self.vel_cmd_offset: Tuple[float, float, float] = (-self.max_x_vel, -self.max_y_vel, -self.max_yaw_rate)
         
        self.time_last_key_up = time.time()
        #stablize
        self.max_x_vel = self.max_x_vel - eps 
        self.max_y_vel = self.max_y_vel - eps
        self.max_yaw_rate = self.max_yaw_rate - eps

    def update(self):
        # Handle Pygame events
        for event in pygame.event.get():
            # Allow exiting the program
            if event.type == pygame.QUIT:
                running = False
                break
            # Handle key press events
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.x_vel = self.max_x_vel
                elif event.key == pygame.K_s:
                    self.x_vel = -self.max_x_vel
                elif event.key == pygame.K_a:
                    self.y_vel = -self.max_y_vel
                elif event.key == pygame.K_d:
                    self.y_vel = self.max_y_vel
                elif event.key == pygame.K_q:
                    self.yaw_vel = self.max_yaw_rate
                elif event.key == pygame.K_e:
                    self.yaw_vel = -self.max_yaw_rate
            # Handle key release events
            elif event.type == pygame.KEYUP:
                self.time_last_key_up = time.time()
                if event.key in (pygame.K_w, pygame.K_s):
                    self.x_vel = 0.0
                elif event.key in (pygame.K_a, pygame.K_d):
                    self.y_vel = 0.0
                elif event.key in (pygame.K_q, pygame.K_e):
                    self.yaw_vel = 0.0
    