from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from nav_gym.nav_legged_gym.envs.local_nav_env_m import LocalNavEnv

    ANY_ENV = Union[LeggedEnv]

import torch

"""
Common termitation checking functions
"""
def time_out(env: "LocalNavEnv", params):
    return env.episode_length_buf >= env.max_episode_length

def command_resample(env: "ANY_ENV", params):
    num_commands = params.get("num_commands", None)
    if num_commands is None:
        return env.command_time_left <= 0.
    return torch.logical_and((env.command_time_left <= 0.), (env.num_commands==num_commands))

def illegal_contact(env: "ANY_ENV", params):
    return torch.any(
        torch.norm(env.robot.net_contact_forces[:, params["body_indices"], :], dim=-1) > 1.0,
        dim=1,
    )

def illegal_force(env: "ANY_ENV", params):
    return torch.any(
        torch.norm(env.robot.net_contact_forces[:, params["body_indices"], :], dim=-1) > params["max_force"],
        dim=1,
    )

def bad_orientation(env: "ANY_ENV", params):
    return torch.acos(-env.robot.projected_gravity_b[:, 2]).abs() > params["limit_angle"]


def torque_limit(env: "ANY_ENV", params):
    return ~torch.all(torch.isclose(env.robot.des_dof_torques, env.robot.dof_torques), dim=1)


def velocity_limit(env: "ANY_ENV", params):
    return torch.any(env.robot.dof_vel.abs() > params["max_vel"], dim=1)


def base_height(env: "ANY_ENV", params):
    return env.robot.root_pos_w[:, 2] < params["limit"]


def dof_pos_limit(env: "ANY_ENV", params):
    out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.0) + (
        env.dof_pos - env.dof_pos_limits[:, 1]
    ).clip(min=0.0)
    return torch.any(out_of_limits > 1.0e-6, dim=1)


