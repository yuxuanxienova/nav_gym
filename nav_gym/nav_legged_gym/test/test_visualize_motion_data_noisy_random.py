from nav_gym.nav_legged_gym.envs.config_locomotion_env import LocomotionEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym import NAV_GYM_ROOT_DIR
from nav_gym.learning.datasets.motion_loader import MotionLoader
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
    env_cfg.terrain_unity.translation = [0.0, 0.0, -0.2]
    env_cfg.terrain_unity.env_origin_pattern = "point"

    env_cfg.gym.viewer.eye = (3.0, 3.0, 3.0)

    env = LocomotionEnv(env_cfg)
    env.set_flag_enable_reset(False)
    env.set_flag_enable_resample(False)
    obs, extras = env.reset()
    
    #Load the Motion Data
    datasets_root = os.path.join(NAV_GYM_ROOT_DIR + "/resources/fld/motion_data/")
    motion_names = ["motion_data_pace1.0.pt","motion_data_walk01_0.5.pt","motion_data_walk03_0.5.pt","motion_data_canter02_1.5.pt"]

    motion_loader = MotionLoader(
        device="cuda",
        file_names=motion_names,
        file_root=datasets_root,
        corruption_level=0.0,
        reference_observation_horizon=2,
        test_mode=False,
        test_observation_dim=None
    )
    motion_idx = 0
    num_motion_clips, num_steps, motion_features_dim = motion_loader.data_list[motion_idx].size()
    # Main loop
    running = True
    sample_length = 16
    max_eps = sample_length * env.cfg.control.decimation
    motion_data_clip = motion_loader.sample_k_steps_from_motion_clip(motion_idx, sample_length)# Shape: [k, motion_features_dim]
    for i in range(max_eps):
        step = int(i // env.cfg.control.decimation) % sample_length
        env.render()
        env.gym.simulate(env.sim) 
        #motion_loader.data_list: (num_files)
        #motion_loader.data_list[0]: [num_motion, num_steps, motion_features_dim]
        
        motion_data_per_step = motion_data_clip[step,:].repeat(env.num_envs, 1)
        #motion_data_per_step: [num_motion, motion_features_dim]
        state_idx_dict = motion_loader.state_idx_dict

        env.robot.dof_pos[:] = motion_loader.get_dof_pos(motion_data_per_step)
        env.robot.dof_vel[:] = motion_loader.get_dof_vel(motion_data_per_step)
        root_pos = motion_loader.get_base_pos(motion_data_per_step)
        root_pos[:, :2] = root_pos[:, :2] + env.terrain.env_origins[:, :2]
        env.robot.root_states[:, :3] = root_pos
        root_ori = motion_loader.get_base_quat(motion_data_per_step)
        env.robot.root_states[:, 3:7] = root_ori
        env.robot.root_states[:, 7:10] = quat_rotate(root_ori, motion_loader.get_base_lin_vel(motion_data_per_step))
        env.robot.root_states[:, 10:13] = quat_rotate(root_ori, motion_loader.get_base_ang_vel(motion_data_per_step))

        env_ids_int32 = torch.arange(env.num_envs, device=env.device).to(dtype=torch.int32)        
        env.gym.set_dof_state_tensor_indexed(env.sim,
                                            gymtorch.unwrap_tensor(env.gym_iface.dof_state),
                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                    gymtorch.unwrap_tensor(env.robot.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        env.gym.refresh_rigid_body_state_tensor(env.sim)
            

