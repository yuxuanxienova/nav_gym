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
    datasets_root = os.path.join(NAV_GYM_ROOT_DIR + "/resources/anymal_d/datasets/record/")
    motion_names = "sampled_amp_observations.pt"
    save_path = os.path.join(datasets_root, motion_names)
    loaded_amp_obs = torch.load(save_path)
    print(f"Loaded AMP observations from {save_path}")
    state_index_dict = {
                            "dof_pos": [0, 12],
                            "dof_vel": [12, 24]
                            }

    # Main loop
    running = True
    sample_length = 16
    max_eps = sample_length * env.cfg.control.decimation
    while running:
        motion_data_clip = loaded_amp_obs[0]# Shape: [sample_length, motion_features_dim]
        for i in range(max_eps):
            step = int(i // env.cfg.control.decimation) % sample_length
            env.render()
            env.gym.simulate(env.sim) 
            motion_data_per_step = motion_data_clip[step].reshape(1, -1).repeat(env.num_envs, 1)# Shape: [num_envs, motion_features_dim]


            env.robot.dof_pos[:] = motion_data_per_step[:, state_index_dict["dof_pos"][0]:state_index_dict["dof_pos"][1]]
            env.robot.dof_vel[:] = motion_data_per_step[:, state_index_dict["dof_vel"][0]:state_index_dict["dof_vel"][1]]
            root_pos = torch.tensor([0.0, 0.0, 3.0], device=env.device).repeat(env.num_envs, 1)
            root_pos[:, :2] = root_pos[:, :2] + env.terrain.env_origins[:, :2]
            env.robot.root_states[:, :3] = root_pos
            root_ori = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
            env.robot.root_states[:, 3:7] = root_ori
            # env.robot.root_states[:, 7:10] = quat_rotate(root_ori, motion_loader.get_base_lin_vel(motion_data_per_step))
            # env.robot.root_states[:, 10:13] = quat_rotate(root_ori, motion_loader.get_base_ang_vel(motion_data_per_step))

            env_ids_int32 = torch.arange(env.num_envs, device=env.device).to(dtype=torch.int32)        
            env.gym.set_dof_state_tensor_indexed(env.sim,
                                                gymtorch.unwrap_tensor(env.gym_iface.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

            env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                        gymtorch.unwrap_tensor(env.robot.root_states),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            

