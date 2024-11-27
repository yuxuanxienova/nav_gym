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
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    policy = runner.get_inference_policy()
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
    max_eps = num_steps * env.cfg.control.decimation

    # Initialize tensor to store feet positions (4 feet * 3 coordinates)
    feet_pos_data = torch.zeros(
        num_motion_clips,
        num_steps,
        12,
        dtype=torch.float,
        device=env.device,
        requires_grad=False,
    )


    for step in range(num_steps):
        # step = int(i // env.cfg.control.decimation) % num_steps
        # env.render()
        # env.gym.simulate(env.sim) 
        action = policy(obs)
        obs, _, _, extras = env.step(action)
        #motion_loader.data_list: (num_files)
        #motion_loader.data_list[0]: [num_motion, num_steps, motion_features_dim]
        motion_data_per_step = motion_loader.data_list[motion_idx][0, step].repeat(env.num_envs, 1)
        #motion_data_i: [num_motion, motion_features_dim]
        state_idx_dict = motion_loader.state_idx_dict

        env.robot.dof_pos[:] = motion_loader.get_dof_pos(motion_data_per_step)
        env.robot.dof_vel[:] = motion_loader.get_dof_vel(motion_data_per_step)
        root_pos = motion_loader.get_base_pos(motion_data_per_step)
        root_pos[:, :2] = root_pos[:, :2] + env.terrain.env_origins[:, :2]
        env.robot.root_states[:, :3] = root_pos
        root_ori = motion_loader.get_base_quat(motion_data_per_step)
        env.robot.root_states[:, 3:7] = root_ori
        env.robot.root_states[:, 7:10] = quat_rotate(root_ori, motion_loader.get_base_lin_vel_b(motion_data_per_step))
        env.robot.root_states[:, 10:13] = quat_rotate(root_ori, motion_loader.get_base_ang_vel_b(motion_data_per_step))

        env_ids_int32 = torch.arange(env.num_envs, device=env.device).to(dtype=torch.int32)        
        env.gym.set_dof_state_tensor_indexed(env.sim,
                                            gymtorch.unwrap_tensor(env.gym_iface.dof_state),
                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                    gymtorch.unwrap_tensor(env.robot.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        # Get feet positions in world frame
        feet_positions_world = env.robot.feet_positions_w # Shape: [1, num_feet, 3]
        # print(feet_positions_world)
        num_feet = feet_positions_world.shape[1]
        # Transform feet positions to robot frame
        # order ['LF_FOOT', 'LH_FOOT', 'RF_FOOT', 'RH_FOOT']
        for feet_idx in range(num_feet):
            feet_i_position_w = feet_positions_world[:,feet_idx,:]
            feet_i_position_b = quat_rotate_inverse(root_ori, feet_i_position_w - root_pos)
            feet_pos_data[0,step,(3*feet_idx):(3*feet_idx+3)] = feet_i_position_b
            
        print(feet_pos_data)

    #Handle last setp
    feet_pos_data[0,num_steps-1,:] = feet_pos_data[0,num_steps-2,:]
    print(feet_pos_data)

    # Append the feet positions to the original motion data
    new_motion_data = torch.cat([motion_loader.data_list[motion_idx], feet_pos_data], dim=2)
    print(f"Original motion data shape: {motion_loader.data_list[motion_idx]}")
    print(f"Feet positions data shape: {feet_pos_data.shape}")
    print(f"New motion data shape: {new_motion_data.shape}")

    # Save the new motion data
    new_motion_name = motion_names[motion_idx].replace(".pt", "_with_feet_pos.pt")
    new_motion_path = os.path.join(datasets_root, new_motion_name)
    torch.save(new_motion_data, new_motion_path)
    print(f"Saved new motion data with feet positions to {new_motion_path}")       

