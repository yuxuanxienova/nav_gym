import os
import time

# Your existing imports
# from legged_gym import LeggedNavEnv, LeggedNavEnvCfg, OnPolicyRunner, TrainConfig, class_to_dict
from nav_gym.nav_legged_gym.envs.config_locomotion_mimic_env import LocomotionMimicEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_mimic_env import LocomotionMimicEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion_mimic import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
import nav_gym.nav_legged_gym.common.rewards.rewards as R
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

    # Set up your environment and policy
    log_dir = os.path.join(os.path.dirname(__file__), "logs/" + time.strftime("%Y%m%d-%H%M%S"))
    # checkpoint_dir = os.path.join(os.path.dirname(__file__), "logs/20241103-092555/" + "model_26099.pt")
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)


    #Override the default config
    env_cfg = LocomotionMimicEnvCfg()
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

    env = LocomotionMimicEnv(env_cfg)
    env.set_flag_enable_reset(False)
    env.set_flag_enable_resample(False)
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    # runner.load(checkpoint_dir)
    policy = runner.get_inference_policy()
    obs, extras = env.reset()

    # Main loop
    running = True
    while running:
        #1. Run the Policy
        action = policy(obs)

        #2. Step the environment
        obs, _, _, extras = env.step(action)

        #3. simulate the shadow env
        #root pos
        env.robot.root_states[idx_shadow_env, 0:3] = env.robot.root_states[idx_main_env, 0:3]
        #root ori
        env.robot.root_states[idx_shadow_env, 3:7] = env.robot.root_states[idx_main_env, 3:7]
        #dof pos
        env.robot.dof_pos[idx_shadow_env] = env.robot.dof_pos[idx_main_env]
        #-----------1. test leg_idx_dict_rel----------------
        # env.robot.dof_pos[idx_shadow_env,env.mimic_module.motion_loader.leg_idx_dict_rel["dof_pos_leg_hl"]] = torch.tensor([1.0,0.0,0.0],device=env.device).reshape(1,3)
        #-----------2. test get_dof_pos_leg_hl_cur_step() [test leg_idx_dict_abs]----------------
        env.robot.dof_pos[idx_shadow_env,env.mimic_module.motion_loader.leg_idx_dict_rel["dof_pos_leg_hr"]] = env.mimic_module.get_dof_pos_leg_hr_cur_step()[idx_main_env]
        #linear velocity#
        env.robot.root_states[idx_shadow_env, 7:10] = env.mimic_module.get_base_lin_vel_w_cur_step()[idx_main_env]
        #angular velocity
        env.robot.root_states[idx_shadow_env, 10:13] = env.mimic_module.get_base_ang_vel_w_cur_step()[idx_main_env]

        env_ids_int32 = torch.arange(env.num_envs, device=env.device).to(dtype=torch.int32)        
        env.gym.set_dof_state_tensor_indexed(env.sim,
                                            gymtorch.unwrap_tensor(env.gym_iface.dof_state),
                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        env.gym.set_actor_root_state_tensor_indexed(env.sim,
                                                    gymtorch.unwrap_tensor(env.robot.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        # env.robot.find_dofs('RF_HAA')#----------------------->6
        # env.robot.find_dofs('LF_HAA')#----------------------->0
        #Print the reward
        print("[INFO]get_euler_xyz(env.robot.root_quat_w):{0}".format(get_euler_xyz(env.robot.root_quat_w)))
        print("[INFO]env.mimic_module.get_dof_pos_cur_step()[0]:{0}".format(env.mimic_module.get_dof_pos_cur_step()[0]))
        print("[INFO]env.robot.dof_pos()[0]:{0}".format(env.robot.dof_pos[0]))
        print("[REWARD]R.mimic_tracking_dof_pos(env,None)[0]:{0}".format(R.mimic_tracking_dof_pos(env,None)[0]))

            

