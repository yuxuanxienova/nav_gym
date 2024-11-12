from nav_gym.nav_legged_gym.envs.config_locomotion_env import LocomotionEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym import NAV_GYM_ROOT_DIR
from nav_gym.nav_legged_gym.utils.motion_processing.data_loader_pickle import MotionLoaderPickle

# isaac-gym
from isaacgym import gymtorch
from isaacgym.torch_utils import (
    quat_rotate,
    quat_rotate_inverse,
    get_euler_xyz,
    quat_from_euler_xyz,
)
from isaacgym import gymtorch, gymapi, gymutil
#python
import torch
import os
import time
PLAY_LOADED_DATA = True
SAVE_DATA = False
SAVE_CORRECT_VEL_DATA = False
MOVE_CAMERA = False
PLOT = False
VIS = True

def generate_target(target_pos, env):
    sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))
    local_target_pos = target_pos - env.terrain.env_origins
    for i in range(env.num_envs):
        x = local_target_pos[i, 0]
        y = local_target_pos[i, 1]
        z = local_target_pos[i, 2]
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[i], sphere_pose)


if __name__ == "__main__":
    log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)

    #Override the default config
    env_cfg = LocomotionEnvCfg()
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.robot.randomization.randomize_friction = False
    env_cfg.randomization.push_robots = False
    env_cfg.terrain_unity.terrain_file = "/terrain/Plane1.obj"
    env_cfg.terrain_unity.translation = [0.0, 0.0, -1.0]
    env_cfg.terrain_unity.env_origin_pattern = "point"

    env_cfg.gym.viewer.eye = (3.0, 3.0, 3.0)

    env = LocomotionEnv(env_cfg)
    env.set_flag_enable_reset(False)
    env.set_flag_enable_resample(False)
    obs, extras = env.reset()
    
    #Load the Motion Data
    state_idx_dict = {
        "base_pos": [0, 1, 2],
        "base_quat": [3, 4, 5, 6],
        "base_lin_vel": [7, 8, 9],
        "base_ang_vel": [10, 11, 12],
        "projected_gravity": [13, 14, 15],
        "dof_pos_leg_fr": [16, 17, 18],
        "dof_pos_leg_fl": [19, 20, 21],
        "dof_pos_leg_hr": [22, 23, 24],
        "dof_pos_leg_hl": [25, 26, 27],
        "dof_vel_leg_fr": [28, 29, 30],
        "dof_vel_leg_fl": [31, 32, 33],
        "dof_vel_leg_hr": [34, 35, 36],
        "dof_vel_leg_hl": [37, 38, 39],
    }
    dim_of_interest = torch.cat([torch.tensor(ids, device=env.device, dtype=torch.long, requires_grad=False) for state, ids in state_idx_dict.items() if ((state != "base_pos") and (state != "base_quat"))])
    datasets_root = os.path.join(NAV_GYM_ROOT_DIR + "/resources/fld/motion_data/")
    motion_name = "motion_data_pace1.0.pt"
    motion_path = os.path.join(datasets_root, motion_name)
    if not os.path.exists(motion_path):
        raise Exception(f"Could not find motion data at {motion_path}")
    motion_loader = MotionLoaderPickle(
        device=env.device,
        motion_file=motion_path,
        dim_of_interest=dim_of_interest,
        state_idx_dict=state_idx_dict,
    )
    loaded_num_trajs, loaded_num_steps, loaded_obs_dim = motion_loader.motion_data.size()
    print(f"[Motion Loader] Loaded motion {motion_name} with {loaded_num_trajs} trajectories, {loaded_num_steps} steps with {loaded_obs_dim} dimensions.")
    max_eps = loaded_num_steps * env.cfg.control.decimation  # 5 * loaded_num_steps * env.cfg.control.decimation # (loaded_num_steps - 50) * env.cfg.control.decimation
    feet_pos_data = torch.zeros(
        loaded_num_trajs, loaded_num_steps, 12, dtype=torch.float, device=env.device, requires_grad=False
    )
    # Main loop
    running = True
    while running:
        for i in range(max_eps):
            env.render()
            env.gym.simulate(env.sim) 
            step = int(i // env.cfg.control.decimation) % loaded_num_steps
            if (step == 0) and (motion_name in ["angvel1.5", "angvelneg1.5", "vely1.5", "velyneg1.5"]):
                step = 1
            # step = i
            env.robot.dof_pos[:] = motion_loader.get_dof_pos()[:, step, :].repeat(env.num_envs, 1) + env.robot.default_dof_pos[:]
            # env.dof_pos[:] = motion_loader.get_dof_pos()[:, step, :].repeat(env.num_envs, 1) + wrong_default_dof_pos[:]
            env.robot.dof_vel[:] = motion_loader.get_dof_vel()[:, step, :].repeat(env.num_envs, 1)
            root_pos = motion_loader.get_base_pos()[:, step, :].repeat(env.num_envs, 1)
            root_pos[:, :2] = root_pos[:, :2] + env.terrain.env_origins[:, :2]
            env.robot.root_states[:, :3] = root_pos
            root_ori = motion_loader.get_base_quat()[:, step, :].repeat(env.num_envs, 1)
            env.robot.root_states[:, 3:7] = root_ori
            base_lin_vel = motion_loader.get_base_lin_vel()[:, step, :].repeat(env.num_envs, 1)
            base_ang_vel = motion_loader.get_base_ang_vel()[:, step, :].repeat(env.num_envs, 1)
            # env.root_states[:, 7:10] = quat_rotate(root_ori, motion_loader.get_base_lin_vel()[:, step, :].repeat(env.num_envs, 1))
            # env.root_states[:, 10:13] = quat_rotate(root_ori, motion_loader.get_base_ang_vel()[:, step, :].repeat(env.num_envs, 1))

            env_ids_int32 = torch.arange(env.num_envs, device=env.device).to(dtype=torch.int32)        
            env.gym.set_dof_state_tensor_indexed(env.sim,
                                                gymtorch.unwrap_tensor(env.gym_iface.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

            env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                        gymtorch.unwrap_tensor(env.robot.root_states),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            # if i % env.cfg.control.decimation == 0:
            #     env.robot.rigid_body_states = env.gym_iface.rigid_body_state.view(env.num_envs, env.robot.num_bodies, 13)
            #     env.robot.feet_positions = env.robot.rigid_body_states[:, env.robot.feet_indices, :3] - env.robot.root_pos_w.unsqueeze(1)
            #     feet_pos_local = torch.zeros_like(env.robot.feet_positions)
            #     for k in range(len(env.robot.feet_indices)):
            #         feet_pos_local[:, k] = quat_rotate_inverse(
            #             env.robot.root_quat_w,
            #             env.robot.feet_positions[:, k]
            #         )
            #         env.robot.feet_positions = feet_pos_local.flatten(1, 2)
            #         feet_pos_data[:, step, :] = env.robot.feet_positions
            # if VIS:
            #     env.gym.clear_lines(env.viewer)
            #     for j in range(4):
            #         generate_target(env.robot.feet_positions[:, env.robot.feet_indices[j], :3], env)
            

