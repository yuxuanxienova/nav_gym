import pygame
import time
class InteractModule:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        # Create a small window to capture keyboard inputs
        screen = pygame.display.set_mode((200, 200))
        pygame.display.set_caption("Robot Controller")

        # Initialize velocities
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.yaw_vel = 0.0

        # Define maximum velocities
        self.max_x_vel = 2.0
        self.max_y_vel = 1.0
        self.max_yaw_vel = 1.0  # Adjust this value based on your environment's limits

        # 
        self.time_last_key_up = time.time()
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
                    self.yaw_vel = self.max_yaw_vel
                elif event.key == pygame.K_e:
                    self.yaw_vel = -self.max_yaw_vel
            # Handle key release events
            elif event.type == pygame.KEYUP:
                self.time_last_key_up = time.time()
                if event.key in (pygame.K_w, pygame.K_s):
                    self.x_vel = 0.0
                elif event.key in (pygame.K_a, pygame.K_d):
                    self.y_vel = 0.0
                elif event.key in (pygame.K_q, pygame.K_e):
                    self.yaw_vel = 0.0
    