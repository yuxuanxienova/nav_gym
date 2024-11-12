import os
import time

# Your existing imports
# from legged_gym import LeggedNavEnv, LeggedNavEnvCfg, OnPolicyRunner, TrainConfig, class_to_dict
from nav_gym.nav_legged_gym.envs.config_locomotion_fld_env import LocomotionFLDEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_fld_env import LocomotionFLDEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.nav_legged_gym.test.interactive_module import InteractModuleVelocity,InteractModulePosition
#isaac
from isaacgym import gymtorch
from isaacgym.torch_utils import (
    quat_rotate,
    quat_rotate_inverse,
    get_euler_xyz,
    quat_from_euler_xyz,
)
#python
import torch
import os
import time
if __name__ == "__main__":
    interactive_module = InteractModuleVelocity()
    # Set up your environment and policy
    log_dir = os.path.join(os.path.dirname(__file__), "logs/" + time.strftime("%Y%m%d-%H%M%S"))
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "logs/20241103-092555/" + "model_26099.pt")
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)


    #Override the default config
    env_cfg = LocomotionFLDEnvCfg()
    env_cfg.env.num_envs = 2
    idx_main_env = [0]
    idx_shadow_env = [1]
    num_shadow_envs = len(idx_shadow_env)
    env_cfg.robot.randomization.randomize_friction = False
    env_cfg.randomization.push_robots = False
    env_cfg.terrain_unity.terrain_file = "/terrain/Plane1.obj"
    env_cfg.terrain_unity.translation = [0.0, 0.0, -1.0]
    env_cfg.terrain_unity.env_origin_pattern = "point"

    env_cfg.gym.viewer.eye = (3.0, 3.0, 3.0)

    env = LocomotionFLDEnv(env_cfg)
    env.set_flag_enable_reset(False)
    env.set_flag_enable_resample(False)
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    runner.load(checkpoint_dir)
    policy = runner.get_inference_policy()
    obs, extras = env.reset()

    # Main loop
    running = True
    while running:
        interactive_module.update()
        
        
        # Run the policy and step the environment
        action = policy(obs)
        obs, _, _, extras = env.step(action)
        #shadow_env
        max_eps = 31
        for step in range(max_eps):
            env.set_velocity_commands(interactive_module.x_vel, interactive_module.y_vel, interactive_module.yaw_vel)
            print("[INFO][COMMAND]x_vel: ", interactive_module.x_vel, "y_vel: ", interactive_module.y_vel, "yaw_vel: ", interactive_module.yaw_vel)
            
            env.render()
            env.gym.simulate(env.sim) 

            env.robot.dof_pos[idx_shadow_env] = env.fld_module.get_reconstructed_dof_pos()[idx_shadow_env, step, :] + env.robot.default_dof_pos[idx_shadow_env]
            # env.robot.dof_vel[:] = motion_loader.get_dof_vel()[:, step, :].repeat(env.num_envs, 1)
            if step == 0:
                #position
                env.robot.root_states[idx_shadow_env, :3] = env.robot.root_states[idx_main_env, :3]
                #orientation
                env.robot.root_states[idx_shadow_env, 3:7] = env.robot.root_states[idx_main_env, 3:7]

            root_ori = env.robot.root_states[idx_shadow_env, 3:7]
            #linear velocity
            base_lin_vel = env.fld_module.get_reconstructed_base_lin_vel()[idx_shadow_env, step, :]
            env.robot.root_states[idx_shadow_env, 7:10] = quat_rotate(root_ori.reshape(num_shadow_envs,-1), base_lin_vel.reshape(num_shadow_envs,-1))
            #angular velocity
            base_ang_vel = env.fld_module.get_reconstructed_base_ang_vel()[idx_shadow_env, step, :]
            env.robot.root_states[idx_shadow_env, 10:13] = quat_rotate(root_ori.reshape(num_shadow_envs,-1), base_ang_vel.reshape(num_shadow_envs,-1))

            env_ids_int32 = torch.arange(env.num_envs, device=env.device).to(dtype=torch.int32)        
            env.gym.set_dof_state_tensor_indexed(env.sim,
                                                gymtorch.unwrap_tensor(env.gym_iface.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

            env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                        gymtorch.unwrap_tensor(env.robot.root_states),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            

