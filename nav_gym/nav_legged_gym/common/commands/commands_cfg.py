from typing import Tuple
from nav_gym.nav_legged_gym.utils.config_utils import configclass


@configclass
class UnifromVelocityCommandCfg:
    class_name: str = "UnifromVelocityCommand"
    robot_name: str = "robot"
    curriculum = False
    max_curriculum = 1.0
    num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw
    resampling_time = 10.0  # time before commands are changed [s]
    heading_command = True  # if true: compute ang vel command from heading error
    prob_standing_envs = 0.05  # percentage of the robots are standing
    prob_heading_envs = 0.5  # percentage of the robots follow heading command (the others follow angular velocity)
    prob_zero_dim = 0.05

    @configclass
    class Ranges:
        lin_vel_x: Tuple = (-1.0, 1.0)  # min max [m/s]
        lin_vel_y: Tuple = (-1.0, 1.0)  # min max [m/s]
        ang_vel_yaw: Tuple = (-1.5, 1.5)  # min max [rad/s]
        heading: Tuple = (-3.14, 3.14)  # [rad]

    ranges = Ranges()