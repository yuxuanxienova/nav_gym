from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
    from nav_gym.nav_legged_gym.envs.local_nav_env import LocalNavEnv
    from nav_gym.nav_legged_gym.envs.locomotion_fld_env import LocomotionFLDEnv
    from nav_gym.nav_legged_gym.envs.locomotion_mimic_env import LocomotionMimicEnv
    from nav_gym.nav_legged_gym.envs.locomotion_pae_env import LocomotionPAEEnv
    from nav_gym.nav_legged_gym.envs.locomotion_pae_latent_scan_residual_command_env import LocomotionPAELatentScanEnv
    from nav_gym.nav_legged_gym.envs.local_nav_pae_latent_scan_residual_command_env import LocalNavPAEEnv

    ANY_ENV = Union[LocomotionEnv]

import torch
from nav_gym.nav_legged_gym.utils.math_utils import wrap_to_pi
"""
Common reward Functions
"""
def feet_acc(env: "LocomotionPAEEnv", params):
    """Penalize the feet acceleration when reaching the goal"""
    return torch.sum(torch.square((env.last_feet_vel - env.robot.feet_vel) / env.dt), dim=1)


def torques(env: "ANY_ENV", params):
    # Penalize torques
    return torch.sum(torch.square(env.robot.dof_torques), dim=1)


def torques_selected(env: "ANY_ENV", params):
    # Penalize torques on selected dofs
    return torch.sum(torch.square(env.robot.dof_torques[:, params["dof_indices"]]), dim=1)


def motor_torques(env: "ANY_ENV", params):
    # Penalize motor torques
    motor_torques = env.robot.dof_torques / env.robot.gear_ratio
    return torch.sum(torch.square(motor_torques), dim=1)


def motor_torques_selected(env: "ANY_ENV", params):
    # Penalize motor torques on selected dofs
    motor_torques = env.robot.dof_torques / env.robot.gear_ratio
    return torch.sum(torch.square(motor_torques[:, params["dof_indices"]]), dim=1)


def dof_vel(env: "ANY_ENV", params):
    # Penalize dof velocities
    return torch.sum(torch.square(env.robot.dof_vel), dim=1)

def dof_vel_selected(env: "ANY_ENV", params):
    # Penalize dof velocities
    return torch.sum(torch.square(env.robot.dof_vel[:, params["dof_indices"]]), dim=1)


def dof_acc(env: "ANY_ENV", params):
    # Penalize dof accelerations
    return torch.sum(torch.square(env.robot.dof_acc), dim=1)

def dof_acc_selected(env: "ANY_ENV", params):
    # Penalize dof accelerations
    return torch.sum(torch.square(env.robot.dof_acc[:, params["dof_indices"]]), dim=1)


def action_rate(env: "ANY_ENV", params):
    # Penalize changes in actions
    return torch.sum(torch.square(env.last_actions - env.actions), dim=1)
def action_rate_2(env: "ANY_ENV", params):
    # Penalize 2nd order changes in actions (technically 2nd order change of last_actions)
    return torch.sum(torch.square(env.last_last_actions - 2.0 * env.last_actions + env.actions), dim=1)

def collision(env: "ANY_ENV", params):
    # Penalize collisions on selected bodies
    collision_force = torch.norm(env.robot.net_contact_forces[:, params["body_indices"], :], dim=-1)
    collision_force = torch.where(collision_force > 1., collision_force.clip(min=200.), collision_force) # minimum penalty of 100
    return torch.sum(collision_force, dim=1) / 200.


def termination(env: "ANY_ENV", params):
    # Terminal reward / penalty
    return env.reset_buf * (1-env.termination_manager.time_out_buf)


def dof_pos_limits(env: "ANY_ENV", params):
    # Penalize dof positions too close to the limit. The soft limit is computed by the env
    out_of_limits = -(env.robot.dof_pos - env.robot.soft_dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
    out_of_limits += (env.robot.dof_pos - env.robot.soft_dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def dof_vel_limits(env: "ANY_ENV", params):
    # Penalize dof velocities too close to the limit
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    # "ratio" defines the soft limit as a percentage of the hard limit
    return torch.sum(
        (torch.abs(env.robot.dof_vel) - env.robot.soft_dof_vel_limits * params["soft_ratio"]).clip(min=0.0, max=1.0),
        dim=1,
    )


def torque_limits(env: "ANY_ENV", params):
    # penalize torques too close to the limit
    # "ratio" defines the soft limit as a percentage of the hard limit
    return torch.sum(
        (torch.abs(env.robot.dof_torques) - env.robot.soft_dof_torque_limits * params["soft_ratio"]).clip(min=0.0),
        dim=1,
    )#-------TODO check using des_dof_torques instead of dof_torques


def contact_forces(env: "ANY_ENV", params):
    # penalize high contact forces
    return torch.sum(
        (torch.norm(env.robot.net_contact_forces, dim=-1) - params["max_contact_force"]).clip(min=0.0),
        dim=1,
    )

def contact_forces_sq(env: "ANY_ENV", params):
    # penalize high contact forces
    contact_forces = torch.norm(env.robot.net_contact_forces, dim=-1).clip(max = 2.*params["max_contact_force"])
    return torch.sum(
        torch.square((contact_forces - params["max_contact_force"]).clip(min=0.0)),
        dim=1,
    )

def mech_power(env: "ANY_ENV", params):
    # penalize high contact forces
    pos_power = (env.robot.dof_torques * env.robot.dof_vel).clip(min=0.0).sum(dim=-1)
    neg_power = (env.robot.dof_torques * env.robot.dof_vel).clip(max=0.0).sum(dim=-1)
    total_power = pos_power + params["recuperation"]*neg_power
    return torch.square(total_power)
    

"""
Locomotion specific reward Functions
"""
def base_motion(env: "LeggedEnv", params):
    body_z_motion = torch.square(env.robot.root_lin_vel_b[:, 2])
    body_ang_motion = torch.sum(torch.square(env.robot.root_ang_vel_b[:, :2]), dim=1)
    rwd_sum = torch.exp(-body_z_motion / params["std_z"]) + torch.exp(-body_ang_motion / params["std_angvel"])
    return rwd_sum

def lin_vel_z(env: "LeggedEnv", params):
    # Penalize z axis base linear velocity
    return torch.square(env.robot.root_lin_vel_b[:, 2])


def ang_vel_xy(env: "LeggedEnv", params):
    # Penalize xy axes base angular velocity
    return torch.sum(torch.square(env.robot.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation(env: "LeggedEnv", params):
    # Penalize non flat base orientation
    return torch.sum(torch.square(env.robot.projected_gravity_b[:, :2]), dim=1)


def vertical_orientation(env: "LeggedEnv", params):
    # Penalize non flat base orientation
    return torch.sum(torch.square(env.robot.projected_gravity_b[:, 1:]), dim=1)


def base_height(env: "LeggedEnv", params):
    # Penalize base height away from target
    # measured_heights = env.sensors[params["sensor"]].get_data()
    measured_heights = env.sensor_manager.get_sensor(params["sensor"]).get_data()
    base_height = torch.mean(env.robot.root_pos_w[:, 2].unsqueeze(1) - measured_heights, dim=1)
    return torch.square(base_height - params["height_target"])

def survival(env: "LeggedEnv", params):
    return torch.ones(env.num_envs, device=env.device).reshape(env.num_envs, )

# def tracking_lin_vel(env: "LeggedEnv", params):
#     # Tracking of linear velocity commands (xy axes)
#     # "std" defines the width of the bel curve
#     lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.robot.root_lin_vel_b[:, :2]), dim=1)
#     return torch.exp(-lin_vel_error / params["std"])


# def tracking_ang_vel(env: "LeggedEnv", params):
#     # Tracking of angular velocity commands (yaw)
#     # "std" defines the width of the bel curve
#     ang_vel_error = torch.square(env.commands[:, 2] - env.robot.root_ang_vel_b[:, 2])
#     return torch.exp(-ang_vel_error / params["std"])
def tracking_lin_vel_x(env: "LeggedEnv", params):
    # Tracking of linear velocity commands (x axes)
    # "std" defines the width of the bel curve
    lin_vel_error = torch.square(env.command_generator.get_velocity_command()[:, 0] - env.robot.root_lin_vel_b[:, 0])
    return torch.exp(-lin_vel_error / params["std"])


def tracking_lin_vel_y(env: "LeggedEnv", params):
    # Tracking of linear velocity commands (y axes)
    # "std" defines the width of the bel curve
    lin_vel_error = torch.square(env.command_generator.get_velocity_command()[:, 1] - env.robot.root_lin_vel_b[:, 1])
    return torch.exp(-lin_vel_error / params["std"])


def tracking_lin_vel(env: "LeggedEnv", params):
    # Tracking of linear velocity commands (xy axes)
    # "std" defines the width of the bel curve
    lin_vel_error = torch.sum(
        torch.square(env.command_generator.get_velocity_command()[:, :2] - env.robot.root_lin_vel_b[:, :2]), dim=1
    )

    return torch.exp(-lin_vel_error / params["std"])

def tracking_lin_vel_direction(env: "LeggedEnv", params):
    cmd_norm = env.command_generator.get_velocity_command()[:, :2].norm(p=2, dim=-1).clamp(min=1e-5)
    cmd_normalized = env.command_generator.get_velocity_command()[:, :2] / cmd_norm.unsqueeze(-1)

    dot_prod = torch.sum(cmd_normalized * env.robot.root_lin_vel_b[:, :2], dim=-1)
    dot_prod[cmd_norm < 0.1] = params["max"] - torch.norm(env.robot.root_lin_vel_b[:, :2])  # stop cmd
    dot_prod = torch.clamp(dot_prod, min=params["min"], max=params["max"])
    return torch.clamp(dot_prod, min=params["min"], max=params["max"])


def tracking_ang_vel(env: "LeggedEnv", params):
    # Tracking of angular velocity commands (yaw)
    # "std" defines the width of the bel curve
    ang_vel_error = torch.square(env.command_generator.get_velocity_command()[:, 2] - env.robot.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / params["std"])

# def feet_air_time(env: "LeggedEnv", params):
#     # Reward long steps
#     first_contact = env.robot.feet_last_air_time > 0.0
#     reward = torch.sum((env.robot.feet_last_air_time - params["time_threshold"]) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.commands[:, :2], dim=1) > 0.1
#     return reward
def feet_air_time(env: "LeggedEnv", params):
    # Reward long steps
    first_contact = env.robot.feet_last_air_time > 0.0
    reward = torch.sum((env.robot.feet_last_air_time - params["time_threshold"]) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_generator.get_velocity_command()[:, :2], dim=1) > 0.1
    return reward

def stumble(env: "LeggedEnv", params):
    # Penalize feet hitting vertical surfaces
    return torch.any(
        torch.norm(env.robot.net_contact_forces[:, :, :2], dim=2)
        > params["hv_ratio"] * torch.abs(env.robot.net_contact_forces[:, :, 2]),
        dim=1,
    )


# def stand_still(env: "LeggedEnv", params):
#     # Penalize motion at zero commands
#     return torch.sum(torch.abs(env.robot.dof_pos - env.robot.default_dof_pos), dim=1) * (
#         torch.norm(env.commands[:, :2], dim=1) < 0.1
#     )
def stand_still(env: "LeggedEnv", params):
    # Penalize motion at zero commands
    return torch.sum(torch.abs(env.robot.dof_pos - env.robot.default_dof_pos), dim=1) * (
        torch.norm(env.command_generator.get_velocity_command()[:, :2], dim=1) < 0.1
    )


def base_motion(env: "LeggedEnv", params):
    body_z_motion = torch.square(env.robot.root_lin_vel_b[:, 2])
    body_ang_motion = torch.sum(torch.square(env.robot.root_ang_vel_b[:, :2]), dim=1)
    rwd_sum = torch.exp(-body_z_motion / params["std_z"]) + torch.exp(-body_ang_motion / params["std_angvel"])
    return rwd_sum

"""
Position tracking rewards
"""
def _command_duration_mask(env: "LeggedEnvPos", duration):
    mask = env.command_time_left <= duration
    return mask / duration


def tracking_pos(env: "LeggedEnvPos", params):
    distance = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1)
    return (1. /(1. + torch.square(distance))) * _command_duration_mask(env, params["duration"])


def tracking_pos2(env: "LeggedEnvPos", params):
    distance = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1)
    return (1 - 0.5*distance)* _command_duration_mask(env, params["duration"])

def tracking_pos3d(env: "LeggedEnvPos", params):
    distance = torch.norm(env.pos_target[:, :3] - env.robot.root_pos_w[:, :3], dim=1)
    return (1 - 0.5*distance)* _command_duration_mask(env, params["duration"])


def tracking_heading(env: "LeggedEnvPos", params):
    distance = wrap_to_pi(env.heading_target - env.robot.heading_w)
    position_distance = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1)
    return (1. /(1. + torch.square(distance))) * (position_distance < params["max_pos_distance"]) * _command_duration_mask(env, params["duration"])

def tracking_heading2(env: "LeggedEnvPos", params):
    distance = torch.abs(wrap_to_pi(env.heading_target - env.robot.heading_w))
    position_distance = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1)
    return (1 - 0.5*distance)* (position_distance < params["max_pos_distance"]) * _command_duration_mask(env, params["duration"])


def dont_wait(env: "LeggedEnvPos", params):
    #TODO add params
    far_away = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1) > 0.25
    waiting = torch.norm(env.robot.root_lin_vel_w[:, :2], dim=1) < params["min_vel"]
    return far_away * waiting


def dont_wait_b(env: "LeggedEnvPos", params):
    far_away = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1) > 0.25
    waiting = torch.norm(env.robot.root_lin_vel_b[:, :2], dim=1) < params["min_vel"]
    return far_away * waiting


def move_in_direction(env: "LeggedEnvPos", params):
    vel_target = env.pos_commands[:, :2] / (torch.norm(env.pos_commands[:, :2], dim=1).unsqueeze(1) + 0.1)
    vel = env.robot.root_lin_vel_b[:, :2] / (torch.norm(env.robot.root_lin_vel_b[:, :2], dim=1).unsqueeze(1) + 0.1)
    return vel[:, 0] * vel_target[:, 0] + vel[:, 1] * vel_target[:, 1] #- self.base_ang_vel[:, 2].abs()


def stand_still_pose(env: "LeggedEnvPos", params):
    should_stand = torch.norm(env.pos_target[:, :2] - env.robot.root_pos_w[:, :2], dim=1) < 0.25
    should_stand &= torch.abs(env.heading_target - env.robot.heading_w) < 0.5
    return torch.sum(torch.square(env.robot.dof_pos - env.robot.default_dof_pos), dim=1) * should_stand * _command_duration_mask(env, params["duration"])


def base_acc(env: "LeggedEnv", params):
    return torch.sum(torch.square((env.robot.last_root_states[:, 7:10] - env.robot.root_states[:, 7:10]) / env.dt), dim=1) + \
        0.02 * torch.sum(torch.square((env.robot.last_root_states[:, 10:13] - env.robot.root_states[:, 10:13]) / env.dt), dim=1)


def feet_acc(env: "LeggedEnv", params):
    return torch.sum(torch.norm(env.robot.body_acc[:, env.robot.feet_indices], dim=-1), dim=1)

"""
ASE specific reward Functions
"""
def tracking_pos_ase(env: "ANY_ENV", params):
    error = torch.sum(torch.square(env.command_generator.get_goal_position_command() - env.robot.root_pos_w),dim=1)
    return torch.exp(-error * params["exp_scale"])



"""
High level specific reward Functions
"""

def _command_duration_mask(env: "LeggedEnvPos", duration):
    mask = env.command_time_left <= duration
    return mask / duration


def termination_hl(env: "LocalNavEnv", params):
    # Terminal reward / penalty
    distance = torch.norm(env.pos_target - env.robot.root_pos_w, dim=1) #.clip(max=4.0)
    distance[distance>4.0] = 4.0
    # return env.reset_buf * ((1-env.termination_manager.time_out_buf) + 0*0.1*distance)
    return env.reset_buf * ((1-env.termination_manager.time_out_buf) + 0.1*distance)


def tracking_pos_hl_final(env: "LocalNavEnv", params):
    distance = torch.norm(env.pos_target - env.robot.root_pos_w, dim=1) #.clip(max=4.0)
    height_diff = torch.abs(env.pos_target[:, 2] - env.robot.root_pos_w[:, 2])
    is_close = (distance < 0.5).float() # 0.3
    distance[distance>4.0] = 4.0
    rew = env.termination_manager.time_out_buf*(20*is_close - distance)
    # rew = (20*is_close - distance)*_command_duration_mask(env, params["duration"])
    # rew = env.termination_manager.time_out_buf*(40*is_close - 0*distance)
    return rew

def tracking_pos_hl(env: "LocalNavEnv", params):
    distance = torch.norm(env.pos_target - env.robot.root_pos_w, dim=1) #.clip(max=4.0)
    distance[distance>50.0] = 50.0
    rew = (1. /(1. + torch.square(distance/20.0)))
    # rew = (20*is_close - distance)*_command_duration_mask(env, params["duration"])
    # rew = env.termination_manager.time_out_buf*(40*is_close - 0*distance)
    return rew

def total_distance_hl_final(env: "LeggedEnvPos", params):
    rew = env.termination_manager.time_out_buf*env.total_distance
    return rew


"""
Local navigation specific reward Functions
"""
def tracking_dense(env: "LeggedEnv", params):
    max_error = params["max_error"]
    total_reward = 1.0 - torch.clip(env.goal_error_z_check, 0.0, max_error) / max_error
    return total_reward

def goal_dot_prod(env: "LeggedEnv", params):
    goal_radius = params["goal_radius"]

    vector_to_goal = env.target_pos - env.robot.root_pos_w
    vector_to_goal = vector_to_goal / torch.norm(vector_to_goal, dim=1).unsqueeze(1)
    dot_prod = torch.sum(vector_to_goal * env.robot.root_lin_vel_w, dim=1)
    total_reward = torch.clip(dot_prod , 0.0, params["max_magnitude"])
    total_reward[env.goal_error_z_check < goal_radius] = params["max_magnitude"]

    return total_reward

def goal_dot_prod_decay(env: "LeggedEnv", params):
    reward = goal_dot_prod(env, params)
    return reward 

def goal_tracking_dense_dot(env: "ANY_ENV", params):
    goal_vector = env.command_generator.get_goal_position_command() - env.robot.root_pos_w
    goal_vector_normalized = goal_vector / torch.norm(goal_vector, dim=1, keepdim=True)
    dot_product = torch.sum(goal_vector_normalized * env.robot.root_lin_vel_w, dim=1)
    # clip such that the maximum magnitude of the dot product is 1
    max_magnitude = params["max_magnitude"]
    dot_product = torch.clamp(dot_product, min=-max_magnitude, max=max_magnitude)

    return dot_product

def reach_goal(env: "ANY_ENV", params):
    goal_radius = params["goal_radius"]
    # Retrieve goal and robot positions as torch tensors
    goal_position = torch.tensor(env.command_generator.get_goal_position_command(), dtype=torch.float32)
    robot_position = torch.tensor(env.robot.root_pos_w, dtype=torch.float32)
    # Calculate the vector from the robot to the goal
    goal_vector = goal_position - robot_position
    distance = torch.norm(goal_vector)  # Euclidean distance to the goal
    # Define reward parameters
    success_reward = params.get("success_reward", 100.0)  # Reward for reaching the goal
    distance_penalty = params.get("distance_penalty", 1.0)  # Penalty per unit distance
    # Calculate the total reward
    if distance < goal_radius:
        total_reward = torch.tensor(success_reward, dtype=torch.float32)  # Significant reward for reaching the goal
    else:
        total_reward = -distance_penalty * distance  # Negative reward proportional to distance
    # Optionally scale the reward
    total_reward = total_reward 
    
    return total_reward

def action_limits_penalty(env: "LeggedEnv", params):
    soft_ratio = params["soft_ratio"]
    vel_limit = env.scaled_action_max * soft_ratio

    return torch.sum(
        (torch.abs(env.scaled_action) - vel_limit).clip(min=0.0),
        dim=1,
    )

def near_goal_stability(env: "ANY_ENV", params):

    total_reward = torch.sum(torch.exp(-torch.square(env.robot.root_states[:, 7:]) / params["std"]), dim=1)
    total_reward[env.goal_error_z_check > params["threshold"]] = 0.0

    return total_reward

def exp_bonus(env: "ANY_ENV", params):
    penalty_max = params["max_count"]

    # Encourage to explore temporally
    # TODO: only valid history
    dist_to_memory_poses = torch.norm(
        env.robot.root_pos_w[:, :2].unsqueeze(1) - env.global_memory.all_graph_nodes_abs[:, :, :2], dim=2
    )
    min_dis = torch.min(dist_to_memory_poses, dim=1)

    ids = torch.linspace(0, env.num_envs - 1, env.num_envs).to(env.device).to(torch.long)
    counts_visited = env.global_memory.all_graph_nodes_abs_counts[ids, min_dis[1]]

    penalty = counts_visited
    penalty_clipped = torch.clip(penalty, 0.0, penalty_max)
    total_reward = 1.0 - penalty_clipped
    return total_reward

def global_exp_volume(env: "ANY_ENV", params):
    # Encourage to explore globally, Naive version, encourage to add new point to global graph
    add_wp_local = env.global_memory.add_wp
    total_rwd = add_wp_local.to(torch.float)
    return total_rwd

def face_front(env: "ANY_ENV", params):
    # robot's moving direction in the cone of the front direction of the robot
    angle_limit = params["angle_limit"]
    min_vel = params["min_vel"]

    lin_vel_norm = torch.norm(env.robot.root_lin_vel_b[:, :2], dim=1)
    lin_vel_direction = env.robot.root_lin_vel_b[:, :2] / (lin_vel_norm.unsqueeze(1) + 1e-6)

    lin_vel_angle_in_base = torch.abs(torch.atan2(lin_vel_direction[:, 1], lin_vel_direction[:, 0]))

    # penalize if the linear velocity angle is larger than the angle limit. range: [0.0, 2.0]
    facing_reward = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)

    # zero if the angle is above the limit
    facing_reward[lin_vel_angle_in_base > angle_limit] = 0.0

    # ignore when small speed.
    facing_reward[lin_vel_norm < min_vel] = 1.0

    return facing_reward
""" FLD specific reward Functions"""

def reward_tracking_reconstructed_lin_vel(env: "LocomotionPAEEnv", params):
    error = torch.sum(torch.square((env.fld_module.target_fld_state - env.fld_module.fld_state)[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["base_lin_vel"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]), dim=1)
    # print("[INFO][reward_tracking_reconstructed_lin_vel][env.fld.latent_idx]{0}".format(env.fld_module.cur_steps))
    # print("[INFO][reward_tracking_reconstructed_lin_vel][self.decoded_obs]{0}".format(env.fld_module.target_fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["base_lin_vel"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_lin_vel][self.fld_state]{0}".format(env.fld_module.fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["base_lin_vel"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_lin_vel]{0}".format(torch.exp(-error * params["exp_scale"])))
    return torch.exp(-error * params["exp_scale"])

def reward_tracking_reconstructed_ang_vel(env: "LocomotionPAEEnv", params):
    error = torch.sum(torch.square((env.fld_module.target_fld_state - env.fld_module.fld_state)[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["base_ang_vel"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]), dim=1)
    # print("[INFO][reward_tracking_reconstructed_ang_vel][env.fld.latent_idx]{0}".format(env.fld_module.cur_steps))
    # print("[INFO][reward_tracking_reconstructed_ang_vel][self.decoded_obs]{0}".format(env.fld_module.target_fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["base_ang_vel"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_ang_vel][self.fld_state]{0}".format(env.fld_module.fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["base_ang_vel"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_ang_vel]{0}".format(torch.exp(-error * params["exp_scale"])))
    return torch.exp(-error * params["exp_scale"])

def reward_tracking_reconstructed_projected_gravity(env: "LocomotionPAEEnv", params):
    error = torch.sum(torch.square((env.fld_module.target_fld_state - env.fld_module.fld_state)[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["projected_gravity"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]), dim=1)
    # print("[INFO][reward_tracking_reconstructed_projected_gravity][env.fld.latent_idx]{0}".format(env.fld_module.cur_steps))
    # print("[INFO][reward_tracking_reconstructed_projected_gravity][self.decoded_obs]{0}".format(env.fld_module.target_fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["projected_gravity"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_projected_gravity][self.fld_state]{0}".format(env.fld_module.fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["projected_gravity"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_projected_gravity]{0}".format(torch.exp(-error * params["exp_scale"])))
    return torch.exp(-error * params["exp_scale"])

def reward_tracking_reconstructed_dof_pos_leg_fl(env: "LocomotionPAEEnv", params):
    error = torch.sum(torch.square((env.fld_module.target_fld_state - env.fld_module.fld_state)[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fl"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]), dim=1)
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fl][env.fld.latent_idx]{0}".format(env.fld_module.cur_steps))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fl][self.decoded_obs]{0}".format(env.fld_module.target_fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fl"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fl][self.fld_state]{0}".format(env.fld_module.fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fl"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fl]{0}".format(torch.exp(-error * params["exp_scale"])))
    return torch.exp(-error * params["exp_scale"])

def reward_tracking_reconstructed_dof_pos_leg_hl(env: "LocomotionPAEEnv", params):
    error = torch.sum(torch.square((env.fld_module.target_fld_state - env.fld_module.fld_state)[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hl"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]), dim=1)
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hl][env.fld.latent_idx]{0}".format(env.fld_module.cur_steps))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hl][self.decoded_obs]{0}".format(env.fld_module.target_fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hl"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hl][self.fld_state]{0}".format(env.fld_module.fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hl"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hl]{0}".format(torch.exp(-error * params["exp_scale"])))
    return torch.exp(-error * params["exp_scale"])

def reward_tracking_reconstructed_dof_pos_leg_fr(env: "LocomotionPAEEnv", params):
    error = torch.sum(torch.square((env.fld_module.target_fld_state - env.fld_module.fld_state)[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fr"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]), dim=1)
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fr][env.fld.latent_idx]{0}".format(env.fld_module.cur_steps))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fr][self.decoded_obs]{0}".format(env.fld_module.target_fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fr"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fr][self.fld_state]{0}".format(env.fld_module.fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_fr"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_fr]{0}".format(torch.exp(-error * params["exp_scale"])))
    return torch.exp(-error * params["exp_scale"])

def reward_tracking_reconstructed_dof_pos_leg_hr(env: "LocomotionPAEEnv", params):
    error = torch.sum(torch.square((env.fld_module.target_fld_state - env.fld_module.fld_state)[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hr"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]), dim=1)
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hr][env.fld.latent_idx]{0}".format(env.fld_module.cur_steps))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hr][self.decoded_obs]{0}".format(env.fld_module.target_fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hr"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hr][self.fld_state]{0}".format(env.fld_module.fld_state[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["dof_pos_leg_hr"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]))
    # print("[INFO][reward_tracking_reconstructed_dof_pos_leg_hr]{0}".format(torch.exp(-error * params["exp_scale"])))
    return torch.exp(-error * params["exp_scale"])

def reward_tracking_reconstructed_feet_pos_fl(env: "LocomotionPAEEnv", params):
    error = torch.sum(torch.square((env.fld_module.target_fld_state - env.fld_module.fld_state)[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["feet_pos_fl"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]), dim=1)
    return torch.exp(-error * params["exp_scale"])

def reward_tracking_reconstructed_feet_pos_fr(env: "LocomotionPAEEnv", params):
    error = torch.sum(torch.square((env.fld_module.target_fld_state - env.fld_module.fld_state)[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["feet_pos_fr"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]), dim=1)
    return torch.exp(-error * params["exp_scale"])

def reward_tracking_reconstructed_feet_pos_hl(env: "LocomotionPAEEnv", params):
    error = torch.sum(torch.square((env.fld_module.target_fld_state - env.fld_module.fld_state)[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["feet_pos_hl"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]), dim=1)
    return torch.exp(-error * params["exp_scale"])

def reward_tracking_reconstructed_feet_pos_hr(env: "LocomotionPAEEnv", params):
    error = torch.sum(torch.square((env.fld_module.target_fld_state - env.fld_module.fld_state)[:, torch.tensor(env.fld_module.target_fld_state_state_idx_dict["feet_pos_hr"], device=env.fld_module.device, dtype=torch.long, requires_grad=False)]), dim=1)
    return torch.exp(-error * params["exp_scale"])
    
"""Mimic specific reward Functions"""

def mimic_tracking_feet_pos_LF(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_feet_pos_b_LF_cur_step() - env.mimic_module.get_robot_feet_pos_b_LF()), dim=1)
    return torch.exp(-error)

def mimic_tracking_feet_pos_LH(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_feet_pos_b_LH_cur_step() - env.mimic_module.get_robot_feet_pos_b_LH()), dim=1)
    return torch.exp(-error)

def mimic_tracking_feet_pos_RF(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_feet_pos_b_RF_cur_step() - env.mimic_module.get_robot_feet_pos_b_RF()), dim=1)
    return torch.exp(-error)

def mimic_tracking_feet_pos_RH(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_feet_pos_b_RH_cur_step() - env.mimic_module.get_robot_feet_pos_b_RH()), dim=1)
    return torch.exp(-error)

def mimic_tracking_dof_pos_fl(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_dof_pos_leg_fl_cur_step() - env.robot.dof_pos[:,env.mimic_module.motion_loader.leg_idx_dict_rel["dof_pos_leg_fl"]]), dim=1)
    return torch.exp(-error)

def mimic_tracking_dof_pos_fr(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_dof_pos_leg_fr_cur_step() - env.robot.dof_pos[:,env.mimic_module.motion_loader.leg_idx_dict_rel["dof_pos_leg_fr"]]), dim=1)
    return torch.exp(-error)

def mimic_tracking_dof_pos_hl(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_dof_pos_leg_hl_cur_step() - env.robot.dof_pos[:,env.mimic_module.motion_loader.leg_idx_dict_rel["dof_pos_leg_hl"]]), dim=1)
    return torch.exp(-error)

def mimic_tracking_dof_pos_hr(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_dof_pos_leg_hr_cur_step() - env.robot.dof_pos[:,env.mimic_module.motion_loader.leg_idx_dict_rel["dof_pos_leg_hr"]]), dim=1)
    return torch.exp(-error)

def mimic_tracking_dof_pos(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_dof_pos_cur_step() - env.robot.dof_pos), dim=1)
    return torch.exp(-error)

def mimic_tracking_dof_vel(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_dof_vel_cur_step() - env.robot.dof_vel), dim=1)
    return torch.exp(-error)

def mimic_tracking_base_lin_vel(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_base_lin_vel_w_cur_step() - env.robot.root_lin_vel_w), dim=1)
    return torch.exp(-error)

def mimic_tracking_base_ang_vel(env: "LocomotionMimicEnv", params):
    error = torch.sum(torch.square(env.mimic_module.get_target_base_ang_vel_w_cur_step() - env.robot.root_ang_vel_w), dim=1)
    return torch.exp(-error)

"""PAE residual specific reward Functions"""
def residual_actions(env: "LocalNavPAEEnv", params):
    sum = torch.sum(torch.square(env.residual_action), dim=1)
    return sum
