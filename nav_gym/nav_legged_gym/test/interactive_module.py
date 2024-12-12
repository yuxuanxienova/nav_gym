import pygame
import time
from typing import Tuple
from nav_gym.nav_legged_gym.envs.config_local_nav_env import LocalNavEnvCfg
import torch
class InteractModuleVelocity:
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
        self.max_y_vel = 0.3
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



class InteractModulePosition:
    def __init__(self, env_cfg:LocalNavEnvCfg=None):
        # Initialize Pygame
        pygame.init()
        # Create a window (you can adjust the size as needed)
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption('Interact Module')

        # Initialize goal positions
        self.x_goal = 0.0
        self.y_goal = 0.0
        self.z_goal = 0.0  # If your environment uses z-coordinate

        # Parameters for position increments
        self.position_increment = 0.1

        # Initialize font for rendering text
        self.font = pygame.font.Font(None, 36)

        # Running flag
        self.running = True

    def update(self):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position
                mouse_pos = pygame.mouse.get_pos()
                # Map mouse position to goal position
                # Adjust these mappings based on your environment's coordinate system
                self.x_goal = (mouse_pos[0] - 200) / 20.0  # Scale factor
                self.y_goal = (200 - mouse_pos[1]) / 20.0  # Invert y-axis and scale

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Reset goal positions
                    self.x_goal = 0.0
                    self.y_goal = 0.0
                    self.z_goal = 0.0

        keys = pygame.key.get_pressed()

        # Update positions based on keys pressed
        if keys[pygame.K_w]:
            self.y_goal += self.position_increment
        if keys[pygame.K_s]:
            self.y_goal -= self.position_increment
        if keys[pygame.K_a]:
            self.x_goal -= self.position_increment
        if keys[pygame.K_d]:
            self.x_goal += self.position_increment
        if keys[pygame.K_q]:
            self.z_goal += self.position_increment
        if keys[pygame.K_e]:
            self.z_goal -= self.position_increment

        # Optionally, limit positions to a maximum value
        max_pos = 10.0
        self.x_goal = max(min(self.x_goal, max_pos), -max_pos)
        self.y_goal = max(min(self.y_goal, max_pos), -max_pos)
        self.z_goal = max(min(self.z_goal, max_pos), -max_pos)

        # Clear the screen
        self.screen.fill((0, 0, 0))  # Fill with black

        # Render the goal positions
        pos_text = f"Goal Position - X: {self.x_goal:.2f}, Y: {self.y_goal:.2f}, Z: {self.z_goal:.2f}"
        text_surface = self.font.render(pos_text, True, (255, 255, 255))  # White text
        self.screen.blit(text_surface, (10, 10))

        # Note: The main loop will handle pygame.display.flip()


class InteractModuleSpaceKey:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        
        # Create a window (you can adjust the size as needed)
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption('Interact Module')
        
        # Running flag
        self.space_pressed = False
        
    def update(self):
        """
        Handles Pygame events and checks for space key presses.
        
        Returns:
            bool: True if the space key was pressed during this update, False otherwise.
        """
        # Initialize the flag for space key press
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.space_pressed = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("Space key pressed")
                    self.space_pressed = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    print("Space key released")
                    self.space_pressed = False
        
        # Clear the screen (fill with black)
        self.screen.fill((0, 0, 0))
        
        # Update the display
        pygame.display.flip()
        

    
    def quit(self):
        """Properly quits Pygame."""
        pygame.quit()

if __name__ == "__main__":
    # Initialize the interactive module
    interact_module = InteractModuleSpaceKey()
    
    # Main loop
    while   True:
        # Update the interactive module
        interact_module.update()
        print("{0}".format(interact_module.space_pressed))
        
        # Update the display
        pygame.display.flip()
    
    # Quit Pygame
    pygame.quit()