
# isaacgym
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse, quat_mul, quat_conjugate, get_euler_xyz, normalize

from isaacgym.torch_utils import to_torch, quat_rotate

# legged_gym
from nav_gym.nav_legged_gym.utils.math_utils import wrap_to_pi
from .commands_cfg import (
    UnifromVelocityCommandCfg,
    WaypointCommandCfg,
)
from nav_gym.nav_legged_gym.utils.warp_utils import ray_cast
# python
from copy import deepcopy
import torch

class CommandBase:
    def __init__(self, cfg, env):
        # prepare some values
        raise NotImplementedError()

    def resample(self, env_ids=None):
        # resample commands
        raise NotImplementedError()

    def update(self, env_ids=None):
        # compute stuff
        raise NotImplementedError()

    def get_goal_position_command(self):
        # returns command data
        raise NotImplementedError()

    def get_velocity_command(self):
        # returns command data
        raise NotImplementedError()

    def set_goal_position_command(self, command):
        # sets command data
        raise NotImplementedError()

    def set_velocity_command(self, command):
        # sets command data
        raise NotImplementedError()

    def reset(self):
        pass

    def log_info(self, env, env_ids, extras_dict):
        pass

class UnifromVelocityCommand(CommandBase):
    def __init__(self, cfg: UnifromVelocityCommandCfg, env: "BaseEnv"):
        self.cfg = cfg
        self.robot = getattr(env, cfg.robot_name)
        self.num_envs = self.robot.num_envs
        self.device = self.robot.device
        self.command_ranges = deepcopy(cfg.ranges)

        # -- command: x vel, y vel, yaw vel, heading
        self.commands = torch.zeros(self.num_envs, self.cfg.num_commands, device=self.device)
        self.tracking_error_sum = torch.zeros(self.num_envs, self.cfg.num_commands, device=self.device)
        self.log_step_counter = torch.zeros(self.num_envs, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
    def update(self, env_ids=None):
        """Sets velocity commands to zero for standing envs, computes angular velocity from heading direction."""

        if self.cfg.heading_command:
            # Compute angular velocity from heading direction for heading envs
            heading_env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            forward = quat_apply(self.robot.root_quat_w[heading_env_ids, :], self.robot._forward_vec_b[heading_env_ids])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[heading_env_ids, 2] = torch.clip(
                0.5 * wrap_to_pi(self.heading_target[heading_env_ids] - heading),
                self.cfg.ranges.ang_vel_yaw[0],
                self.cfg.ranges.ang_vel_yaw[1],
            )

        # Enforce standing (i.e., zero velocity commands) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.commands[standing_env_ids, :] = 0.0

        self.log_data()
    def log_data(self):
        # logs data
        self.tracking_error_sum[:, :2] += torch.abs(self.commands[:, :2] - self.robot.root_lin_vel_b[:, :2])
        self.tracking_error_sum[:, 2] += torch.abs(self.commands[:, 2] - self.robot.root_ang_vel_b[:, 2])
        self.log_step_counter += 1
    def resample(self, env_ids=None):
        """Randomly select commands of some environments."""
        if len(env_ids) == 0:
            return
        print("[INFO]CommandGenerator: Resampling commands for envs: ", env_ids)
        # set tracking error to zero
        self.tracking_error_sum[env_ids] = 0.0
        self.log_step_counter[env_ids] = 0.0

        # resample velocities
        self.resample_velocities(env_ids)
    def resample_velocities(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        # print(self.commands[env_ids], env_ids)
        self.commands[env_ids, 0] = r.uniform_(self.command_ranges.lin_vel_x[0], self.command_ranges.lin_vel_x[1])
        # linear velocity - y direction
        self.commands[env_ids, 1] = r.uniform_(self.command_ranges.lin_vel_y[0], self.command_ranges.lin_vel_y[1])
        # # ang vel yaw - rotation around z
        self.commands[env_ids, 2] = r.uniform_(self.command_ranges.ang_vel_yaw[0], self.command_ranges.ang_vel_yaw[1])
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(self.command_ranges.heading[0], self.command_ranges.heading[1])
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.prob_heading_envs

        # zero dims
        zero_x = r.uniform_(0.0, 1.0) <= self.cfg.prob_zero_dim
        self.commands[env_ids][zero_x, 0] = 0.0
        zero_y = r.uniform_(0.0, 1.0) <= self.cfg.prob_zero_dim
        self.commands[env_ids][zero_y, 1] = 0.0
        zero_z = r.uniform_(0.0, 1.0) <= self.cfg.prob_zero_dim
        self.commands[env_ids][zero_z, 2] = 0.0

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.prob_standing_envs
    def set_velocity_command(self, command):
        """
        Sets the velocity commands for all environments.

        Args:
            x_vel (float or torch.Tensor): Desired x-axis linear velocity.
            y_vel (float or torch.Tensor): Desired y-axis linear velocity.
            yaw_vel (float or torch.Tensor): Desired yaw angular velocity.
        """
        print("[INFO]CommandGenerator: Set Velocity Commands")
        x_vel, y_vel, yaw_vel = command
        # Ensure inputs are tensors
        if not isinstance(x_vel, torch.Tensor):
            x_vel = torch.tensor(x_vel, device=self.device)
        if not isinstance(y_vel, torch.Tensor):
            y_vel = torch.tensor(y_vel, device=self.device)
        if not isinstance(yaw_vel, torch.Tensor):
            yaw_vel = torch.tensor(yaw_vel, device=self.device)
        
        # Expand to [num_envs] if inputs are scalars
        if x_vel.dim() == 0:
            x_vel = x_vel.expand(self.num_envs)
        if y_vel.dim() == 0:
            y_vel = y_vel.expand(self.num_envs)
        if yaw_vel.dim() == 0:
            yaw_vel = yaw_vel.expand(self.num_envs)
        
        # Assign commands using broadcasting
        self.commands[:, 0] = x_vel
        self.commands[:, 1] = y_vel
        self.commands[:, 2] = yaw_vel
    def get_velocity_command(self):
        return self.commands
    def log_info(self, env, env_ids, extras_dict):
        extras_dict["x_vel_tracking_error"] = torch.mean(
            self.tracking_error_sum[env_ids, 0] / env.command_generator.log_step_counter[env_ids]
        )
        extras_dict["y_vel_tracking_error"] = torch.mean(
            self.tracking_error_sum[env_ids, 1] / env.command_generator.log_step_counter[env_ids]
        )
        extras_dict["yaw_vel_tracking_error"] = torch.mean(
            self.tracking_error_sum[env_ids, 2] / env.command_generator.log_step_counter[env_ids]
        )


class WaypointCommand(CommandBase):
    def __init__(self, cfg: WaypointCommandCfg, env: "BaseEnv"):
        self.cfg = cfg
        self.robot = getattr(env, cfg.robot_name)
        self.terrain = env.terrain
        self.num_envs = self.robot.num_envs
        self.device = self.robot.device
        self.command_ranges = deepcopy(cfg.ranges)

        # -- command: x vel, y vel, yaw vel, heading
        self.velocity_commands = torch.zeros(self.num_envs, self.cfg.num_velocity_commands, device=self.device)

        # -- command: x, y, z position
        self.goal_commands = torch.zeros(self.num_envs, self.cfg.num_goal_commands, device=self.device)
        self.tracking_error_sum = torch.zeros(self.num_envs, self.cfg.num_velocity_commands, device=self.device)
        self.log_step_counter = torch.zeros(self.num_envs, device=self.device)

    def get_goal_position_command(self):
        return self.goal_commands[:, :3]

    def get_velocity_command(self):
        return self.velocity_commands

    def set_goal_position_command(self, command):
        self.goal_commands = command

    def set_velocity_command(self, command):
        self.velocity_commands = command

    def resample(self, env_ids=None):
        """Randomly select commands of some environments."""
        if len(env_ids) == 0:
            return

        # set tracking error to zero
        self.tracking_error_sum[env_ids] = 0.0
        self.log_step_counter[env_ids] = 0.0

        # resample velocities
        self.resample_goals(env_ids)

    def resample_goals(self, env_ids):
        """Randomly select commands of some environments."""
        goal_dir = torch.empty(len(env_ids), device=self.device).uniform_(
            self.command_ranges.heading_range[0], self.command_ranges.heading_range[1]
        )
        goal_dist = torch.empty(len(env_ids), device=self.device).uniform_(
            self.command_ranges.radius_range[0], self.command_ranges.radius_range[1]
        )
        self.goal_commands[env_ids, 0] = torch.cos(goal_dir) * goal_dist
        self.goal_commands[env_ids, 1] = torch.sin(goal_dir) * goal_dist
        self.goal_commands[env_ids, :2] += self.robot.root_pos_w[env_ids, :2]

        ray_starts_world = self.goal_commands[env_ids]
        ray_starts_world[:, 2] = 10.0
        ray_directions = torch.zeros_like(ray_starts_world)
        ray_directions[..., :] = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        ray_hit_positions, _ = ray_cast(ray_starts_world, ray_directions, self.terrain.wp_meshes)

        self.goal_commands[env_ids, 2] = ray_hit_positions[:, 2] + 0.5

    def update(self, env_ids=None):
        self.log_data()

    def log_data(self):
        # logs data
        self.tracking_error_sum[:, :2] += torch.abs(self.velocity_commands[:, :2] - self.robot.root_lin_vel_b[:, :2])
        self.tracking_error_sum[:, 2] += torch.abs(self.velocity_commands[:, 2] - self.robot.root_ang_vel_b[:, 2])
        self.log_step_counter += 1
