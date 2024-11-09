from typing import Tuple
from nav_gym.nav_legged_gym.utils.config_utils import configclass


@configclass
class UnifromVelocityCommandCfg:
    class_name: str = "UnifromVelocityCommand"
    robot_name: str = "robot"
    curriculum = False
    max_curriculum = 1.0
    num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw
    resampling_time = 15.0  # time before commands are changed [s]
    heading_command = False  # if true: compute ang vel command from heading error
    prob_standing_envs = 0.05  # percentage of the robots are standing
    prob_heading_envs = 0.5  # percentage of the robots follow heading command (the others follow angular velocity)
    prob_zero_dim = 0.05

    @configclass
    class Ranges:
        lin_vel_x: Tuple = (-2.0, 2.0)  # min max [m/s]
        lin_vel_y: Tuple = (-1.0, 1.0)  # min max [m/s]
        ang_vel_yaw: Tuple = (-1.5, 1.5)  # min max [rad/s]
        heading: Tuple = (-3.14, 3.14)  # [rad]

    ranges = Ranges()

@configclass
class WaypointCommandCfg:
    class_name: str = "WaypointCommand"
    robot_name: str = "robot"
    resampling_time: float = 10.0  # time before commands are changed [s]
    resampling_prob: float = 0.1  # probability of resampling a new waypoint
    num_goal_commands = 3  # default: x, y, z position
    num_velocity_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw

    class Ranges:
        radius_range: Tuple = (4.0, 7.0)  # min max dist [m]
        heading_range: Tuple = (-3.14, 3.14)  # [rad]
        max_velocity: Tuple = (2.0, 1.0, 1.5)

    ranges = Ranges()