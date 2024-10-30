# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from nav_gym.nav_legged_gym.envs.legged_nav_env import LeggedNavEnv

    ANY_ENV = Union[LeggedNavEnv]

import torch
from nav_gym.nav_legged_gym.utils.math_utils import quat_rotate_inverse, yaw_quat
""" Common observation functions
"""


def dof_pos(env: "ANY_ENV", params):
    return env.robot.dof_pos - env.robot.default_dof_pos


def dof_pos_selected(env: "ANY_ENV", params):
    indices = params["dof_indices"]
    return env.robot.dof_pos[indices] - env.robot.default_dof_pos[indices]


def dof_vel(env: "ANY_ENV", params):
    return env.robot.dof_vel


def dof_torques(env: "ANY_ENV", params):
    return env.robot.des_dof_torques


def dof_pos_abs(env: "ANY_ENV", params):
    return env.robot.dof_pos


def actions(env: "ANY_ENV", params):
    return env.actions


def ray_cast(env: "ANY_ENV", params):
    sensor = env.sensor_manager.get_sensor(params["sensor"])
    heights = env.robot.root_pos_w[:, 2].unsqueeze(1) - 0.5 - sensor.get_data()[..., 2]
    return heights


def ray_cast_front(env: "ANY_ENV", params):
    sensor = env.sensors[params["sensor"]]
    sensor.get_data()
    dists = torch.norm(env.robot.root_pos_w[:, :2].unsqueeze(1) - sensor.ray_hits_world[..., :2], dim=-1).clip(0.0, 2.0)
    return dists


def ray_cast_up(env: "ANY_ENV", params):
    sensor = env.sensors[params["sensor"]]
    heights = sensor.get_data()[..., 2] - env.robot.root_pos_w[:, 2].unsqueeze(1) + 0.5
    return heights

# def height_scan(env: "ANY_ENV", params):
#     sensor = env.sensors[params["sensor"]]
#     offset = sensor.cfg.attachement_pos[2]
#     height = sensor.get_distances() - offset
#     return height.reshape(env.num_envs, -1)
def height_scan(env: "ANY_ENV", params):
    sensor = env.sensor_manager.get_sensor(params["sensor"])
    offset = sensor.cfg.attachement_pos[2]
    height = sensor.get_distances() - offset
    return height.reshape(env.num_envs, -1)

def point_cloud(env: "ANY_ENV", params):
    sensor = env.sensor_manager.get_sensor(params["sensor"])
    offset = sensor.cfg.attachement_pos
    # Convert the tuple to a Tensor and move it to the same device as sensor data
    sensor_pos_w =(env.robot.root_pos_w + torch.tensor(offset, device=sensor.get_data().device)).reshape(env.num_envs, 1, -1)#Dim:(num_envs,1, 3)
    return (sensor.get_data() - sensor_pos_w).reshape(env.num_envs, -1)#Dim:(num_envs, n_points, 3)

def imu_acc(env: "ANY_ENV", params):
    sensor = env.sensors[params["sensor"]]
    return sensor.get_data()[:, :3]


def imu_ang_vel(env: "ANY_ENV", params):
    sensor = env.sensors[params["sensor"]]
    return sensor.get_data()[:, 3:6]

""" Locomotion specific observation functions"""


def projected_gravity(env: "ANY_ENV", params):
    return env.robot.projected_gravity_b


def base_lin_vel(env: "ANY_ENV", params):
    return env.robot.root_lin_vel_b

def base_lin_vel2(env: "ANY_ENV", params):
    vel_w = (env.robot.root_states[:, :3] - env.robot.last_root_states[:, :3]) / env.dt
    vel_b = quat_rotate_inverse(env.robot.root_quat_w, vel_w)
    return vel_b

def base_ang_vel(env: "ANY_ENV", params):
    return env.robot.root_ang_vel_b


# def velocity_commands(env: "LeggedEnv", params):
#     return env.commands[:, :3]
def velocity_commands(env: "ANY_ENV", params):
    return env.command_generator.get_velocity_command()

def latent(env: "LeggedEnv", params):
    sensor = env.sensors[params["sensor"]]
    return sensor.get_data()


# specific to pos targets
def pos_commands(env: "LeggedEnvPos", params):
    if env.cfg.commands.override:
        target_vec = torch.tensor([2.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
        env.pos_commands[:] = quat_rotate_inverse(yaw_quat(env.robot.root_quat_w), target_vec)
    return env.pos_commands[:, :2]


def heading_commands(env: "LeggedEnvPos", params):
    return env.heading_commands[:].unsqueeze(1)
    # angle = (env.heading_target - env.robot.heading_w).unsqueeze(1)
    # return torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)

def heading_commands_sin(env: "LeggedEnvPos", params):
    # return env.heading_commands[:].unsqueeze(1)
    angle = (env.heading_target - env.robot.heading_w).unsqueeze(1)
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)


def time_to_target(env: "LeggedEnvPos", params):
    if env.cfg.commands.override:
        return torch.ones(env.num_envs, 1, device=env.device) * 0.2 #25
    return env.command_time_left.unsqueeze(1) / env.cfg.commands.resampling_time[1]


def should_stand(env: "LeggedEnvPos", params):
    should_stand = torch.norm(env.pos_target - env.robot.root_pos_w, dim=1) < 0.5
    should_stand &= torch.abs(env.heading_target - env.robot.heading_w) < 1.
    return 1.* should_stand.unsqueeze(1)

"""Fusing policies"""
def expert_outputs_fuse(env: "LeggedEnvPosFuse", params):
    ### assert env has a tensor called env.expert_outputs of shape (n_env, n_expert, n_actions)
    if not env._init_done:
        print('multi expert outputs not initialized, using zeros')
        return torch.zeros(env.num_envs, env.cfg.env.num_experts * env.num_actions, device=env.device)
    else:
        return env.expert_outputs.reshape(env.num_envs,-1)