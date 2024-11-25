# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
    from nav_gym.nav_legged_gym.envs.local_nav_env import LocalNavEnv
    from nav_gym.nav_legged_gym.envs.locomotion_fld_env import LocomotionFLDEnv
    from nav_gym.nav_legged_gym.envs.locomotion_mimic_env import LocomotionMimicEnv

    ANY_ENV = Union[LocomotionEnv]

import torch
from nav_gym.nav_legged_gym.utils.math_utils import quat_rotate_inverse, yaw_quat
""" Common observation functions
"""


def dof_pos(env: "ANY_ENV", params):
    return env.robot.dof_pos - env.robot.default_dof_pos

def dof_pos_history(env: "ANY_ENV", params):
    #(num_envs, history_length, num_dof)
    return env.dof_pos_history 

def dof_pos_selected(env: "ANY_ENV", params):
    indices = params["dof_indices"]
    return env.robot.dof_pos[indices] - env.robot.default_dof_pos[indices]

def dof_pos_history_selected(env: "ANY_ENV", params):
    indices = params["dof_indices"]
    hist_index = params["hist_index"]
    return env.dof_pos_history[:, hist_index, indices]

def dof_vel(env: "ANY_ENV", params):
    return env.robot.dof_vel

def dof_vel_history(env: "ANY_ENV", params):
    #(num_envs, history_length, num_dof)
    return env.dof_vel_history
def dof_vel_history_selected(env: "ANY_ENV", params):
    indices = params["dof_indices"]
    hist_index = params["hist_index"]

    return env.dof_vel_history[:, hist_index, indices]

def dof_torques(env: "ANY_ENV", params):
    return env.robot.des_dof_torques


def dof_pos_abs(env: "ANY_ENV", params):
    return env.robot.dof_pos


def actions(env: "ANY_ENV", params):
    return env.actions
def last_actions(env: "ANY_ENV", params):
    return env.last_actions
def last_last_actions(env: "ANY_ENV", params):
    return env.last_last_actions
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
    
"""High Level Observation Functions"""
def position_target(env: "LocalNavEnv", params):
    # print("[Debug]Observation: position_target{0}".format(env.pos_target - env.robot.root_pos_w))
    return env.pos_target - env.robot.root_pos_w

def llc_obs(env: "LocalNavEnv", params):
    llc_obs_dict = env.ll_env.obs_manager.compute_obs(env.ll_env)
    obs_to_get = llc_obs_dict[params["name"]]
    return obs_to_get

def height_trav_map(env: "ANY_ENV", params):

    height_sensor = env.sensors[params["sensor"]]

    height_map = height_sensor.get_data()[..., 2] - env.robot.root_pos_w[:, 2].unsqueeze(1)
    max_height = params["max_height"]
    min_height = params["min_height"]

    traversability_map = env.traversability_map[params["sensor"]].clone()

    traversability_map = env.ll_env.traversability_scales * traversability_map
    traversability_map = env.ll_env.traversability_offsets + traversability_map

    # Occlusion handling
    occlusion = (height_map > max_height) | (height_map < min_height)  # this includes unhit rays

    height_map[occlusion] = params["occlusion_fill_height"]
    traversability_map[occlusion] = params["occlusion_fill_travers"]

    # height_map = torch.clip(height_map, min_height,  max_height)

    # traversability as a second channel
    stacked_map = torch.stack([height_map, traversability_map], dim=1)
    return stacked_map

def wp_pos_history(env: "ANY_ENV", params):
    decimation = params["decimation"]
    num_obs = params["num_points"]
    clip_range = params["clip_range"]
    ob_dim_single = 6
    # ob_dim_single = 4
    output = torch.zeros([env.num_envs, num_obs,  ob_dim_single], device=env.device)

    wp_history_data = env.wp_history.history_pos.clone()
    pose_history_data = env.pose_history_exp.history_pos.clone()


    for i in range(num_obs):
        idx = i * decimation
        pos = 0

        # history_pos_xy = pose_history_data[:, idx, :2]
        # wp_history_xy = wp_history_data[:, idx, :2]
        output[:, i, pos: pos + 3] = torch.clamp(pose_history_data[:, idx], -clip_range,clip_range)
        # output[:, i, pos: pos + 2] = max_dist_clip(history_pos_xy, clip_range)
        pos += 3
        output[:, i, pos: pos + 3] = torch.clamp(wp_history_data[:, idx], -clip_range, clip_range)
        # output[:, i, pos: pos + 2] = max_dist_clip(wp_history_xy, clip_range)
        pos += 3

    return output
def node_positions_times(env: "ANY_ENV", params):
    graph_poses_data = env.global_memory.all_graph_nodes.clone()
    graph_poses_data = graph_poses_data[:, : params["num_points"], :]

    # graph_quats_data = env.global_memory.all_graph_nodes_quat.clone()
    # graph_quats_data = graph_quats_data[:, : params["num_points"], :]
    # _, _, yaws = get_euler_xyz(graph_quats_data.reshape(-1, 4))
    # yaws = yaws.reshape(graph_quats_data.shape[0], graph_quats_data.shape[1], 1)

    graph_counts_data = env.global_memory.all_graph_nodes_abs_counts.unsqueeze(-1).clone()
    graph_counts_data = graph_counts_data[:, : params["num_points"], :] * env.dt

    ub = params["counter_limit"]
    graph_counts_data[graph_counts_data > ub] = ub

    graph_data = torch.cat((graph_poses_data, graph_counts_data), dim=2)

    # set zero for nodes that are not used
    num_nodes = env.global_memory.num_nodes
    range_tensor = torch.arange(params["num_points"], device=env.device).repeat(num_nodes.shape[0], 1)
    graph_data[range_tensor >= num_nodes.unsqueeze(1)] = 0.0

    return graph_data

#-----------------------FLD Observation Functions-----------------------
def fld_latent_phase_sin(env: "LocomotionFLDEnv", params):
    return torch.sin(2 * torch.pi * env.fld_module.latent_encoding[:, :, 0])
def fld_latent_phase_cos(env: "LocomotionFLDEnv", params):
    return torch.sin(2 * torch.pi * env.fld_module.latent_encoding[:, :, 0])
def fld_latent_others(env: "LocomotionFLDEnv", params):
    return (env.fld_module.latent_encoding[:, :, 1:].swapaxes(1, 2).flatten(1, 2) - env.fld_module.latent_param_mean) / env.fld_module.latent_param_std

def fld_reconstructed_base_lin_vel(env: "LocomotionFLDEnv", params):
    return env.fld_module.get_reconstructed_base_lin_vel().flatten(1, 2)
def fld_reconstructed_base_ang_vel(env: "LocomotionFLDEnv", params):
    return env.fld_module.get_reconstructed_base_ang_vel().flatten(1, 2)
def fld_reconstructed_projected_gravity(env: "LocomotionFLDEnv", params):
    return env.fld_module.get_reconstructed_projected_gravity().flatten(1, 2)
def fld_reconstructed_dof_pos(env: "LocomotionFLDEnv", params):
    return env.fld_module.get_reconstructed_dof_pos().flatten(1, 2)

#-------------------------- Mimiv Module Functions --------------------------
def mimic_dof_pos_cur_step(env: "LocomotionMimicEnv", params):
    return env.mimic_module.get_dof_pos_cur_step()
def mimic_dof_vel_cur_step(env: "LocomotionMimicEnv", params):
    return env.mimic_module.get_dof_vel_cur_step()
def mimic_base_lin_vel_w_cur_step(env: "LocomotionMimicEnv", params):
    return env.mimic_module.get_base_lin_vel_w_cur_step()
def mimic_base_ang_vel_w_cur_step(env: "LocomotionMimicEnv", params):
    return env.mimic_module.get_base_ang_vel_w_cur_step()

def error_mimic_tracking_dof_pos_fl(env: "LocomotionMimicEnv", params):
    error = env.mimic_module.get_dof_pos_leg_fl_cur_step() - env.robot.dof_pos[:,env.mimic_module.motion_loader.leg_idx_dict_rel["dof_pos_leg_fl"]]
    return error

def error_mimic_tracking_dof_pos_fr(env: "LocomotionMimicEnv", params):
    error = env.mimic_module.get_dof_pos_leg_fr_cur_step() - env.robot.dof_pos[:,env.mimic_module.motion_loader.leg_idx_dict_rel["dof_pos_leg_fr"]]
    return error

def error_mimic_tracking_dof_pos_hl(env: "LocomotionMimicEnv", params):
    error = env.mimic_module.get_dof_pos_leg_hl_cur_step() - env.robot.dof_pos[:,env.mimic_module.motion_loader.leg_idx_dict_rel["dof_pos_leg_hl"]]
    return error

def error_mimic_tracking_dof_pos_hr(env: "LocomotionMimicEnv", params):
    error = env.mimic_module.get_dof_pos_leg_hr_cur_step() - env.robot.dof_pos[:,env.mimic_module.motion_loader.leg_idx_dict_rel["dof_pos_leg_hr"]]
    return error

def error_mimic_tracking_base_lin_vel(env: "LocomotionMimicEnv", params):
    error = env.mimic_module.get_base_lin_vel_w_cur_step() - env.robot.root_lin_vel_w
    return error

def error_mimic_tracking_base_ang_vel(env: "LocomotionMimicEnv", params):
    error = env.mimic_module.get_base_ang_vel_w_cur_step() - env.robot.root_ang_vel_w
    return error