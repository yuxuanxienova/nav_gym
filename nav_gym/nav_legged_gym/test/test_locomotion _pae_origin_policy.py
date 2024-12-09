import os
import time
import pygame

# Your existing imports
# from legged_gym import LeggedNavEnv, LeggedNavEnvCfg, OnPolicyRunner, TrainConfig, class_to_dict
from nav_gym.nav_legged_gym.envs.config_locomotion_pae_env import LocomotionPAEEnvCfg
from nav_gym.nav_legged_gym.test.pae_origin_policy.wild_anymal_d import WildAnymal
from nav_gym.nav_legged_gym.envs.locomotion_pae_env import LocomotionPAEEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion_pae import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym import NAV_GYM_ROOT_DIR
from nav_gym.nav_legged_gym.test.pae_origin_policy.actor_critic import ActorCritic
from nav_gym.nav_legged_gym.test.pae_origin_policy.WildAnymalDPPOCfg import WildAnymalDPPOCfg
import torch
import os
import time
import random
import numpy as np
from isaacgym import gymtorch
if __name__ == "__main__":
    #--------------------------
    # set seed
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # --------------------------
    # Set up your environment and policy
    log_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/" + time.strftime("%Y%m%d-%H%M%S"))
    checkpoint_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/locomotion_pae/20241203-235826/" + "model_2100.pt")
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)

    env_cfg = LocomotionPAEEnvCfg()
    env_cfg.env.num_envs = 1
    # env_cfg.robot.randomization.randomize_friction = False
    # env_cfg.randomization.push_robots = False
    env_cfg.terrain_unity.terrain_file = "/terrain/Plane1.obj"
    env_cfg.terrain_unity.translation = [0.0, 0.0, -1.0]
    env_cfg.terrain_unity.env_origin_pattern = "point"
    env_cfg.gym.viewer.eye = (3.0, 3.0, 3.0)
    #-----------------------
    env = LocomotionPAEEnv(env_cfg)
    env.set_flag_enable_reset(False)
    env.set_flag_enable_resample(False)
    #-------------------
    # env = WildAnymal(env_cfg)
    # env.set_flag_enable_reset(False)
    # env.set_flag_enable_resample(False)

    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    obs, extras = env.reset()
    
    train_cfg = WildAnymalDPPOCfg()
    train_cfg_dict = class_to_dict(train_cfg)
    actor_critic = ActorCritic(runner.num_obs, runner.num_obs, env.num_actions, **train_cfg_dict["policy"]).to(env.device)
    path = "/home/yuxuan/isaac_ws/nav_gym/nav_gym/nav_legged_gym/test/pae_origin_policy/model_8000.pt"
    loaded_dict = torch.load(path)
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    actor_critic.eval()
    policy = actor_critic.act_inference

    #------------Debug:set dof pos manually----------------
    # env.robot.dof_pos[0] = 
    # env_ids_int32 = torch.arange(env.num_envs, device=env.device).to(dtype=torch.int32)        
    # env.gym.set_dof_state_tensor_indexed(env.sim,
    #                                     gymtorch.unwrap_tensor(env.gym_iface.dof_state),
    #                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    # env.gym.refresh_rigid_body_state_tensor(env.sim)
    #-----------------------------------------------
    #Main loop
    while True:
        # Run the policy and step the environment
        #----------Debug: set obs manually------------
        # obs = torch.zeros_like(obs).to(env.device)
        # obs = torch.tensor([[ 5.6983e-04,  5.6411e-04,  1.1727e-03,  2.9305e-04, -2.3627e-04,
        #  -1.6511e-05, -1.1276e-01,  3.9839e-02, -9.9283e-01, -2.7601e-01,
        #   1.6666e-01, -3.7925e-01, -1.0270e-01, -8.8557e-01,  5.2881e-01,
        #   1.2565e-01,  4.7713e-01, -4.1067e-01,  2.8881e-01, -3.8902e-01,
        #   8.1837e-01, -1.8418e-03,  1.2719e-03, -4.4508e-03, -1.2810e-03,
        #   8.3365e-04,  2.3337e-04,  4.2117e-04,  1.8016e-03, -2.8430e-03,
        #   2.5668e-03, -5.0268e-03,  1.0765e-02, -2.3345e-01,  6.7116e-01,
        #  -1.0732e+00, -3.2977e-01, -1.0705e+00,  1.0725e+00,  2.7687e-01,
        #   9.2961e-01, -9.4733e-01,  3.4452e-01, -6.7376e-01,  1.4983e+00,
        #   6.7184e-01, -9.7770e-01, -9.6356e-01,  3.6701e-01,  7.4070e-01,
        #  -2.0999e-01,  2.6750e-01, -9.3022e-01,  1.0000e+00,  0.0000e+00,
        #   0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
        #   0.0000e+00,  0.0000e+00,  0.0000e+00]], device='cuda:0')
        default_dof_pos = torch.tensor([-0.1386,  0.4809, -0.7614, -0.1386, -0.4809,  0.7614,  0.1386,  0.4809,
        -0.7614,  0.1386, -0.4809,  0.7614], device=env.device)
  
        #------------------------------
        action = policy(obs)

        #---------------debug---------------
        # print("obs", obs)
        # print("actions", action)
        # print("env.fld_module.latent_encoding[:, :, 0]",env.fld_module.latent_encoding[:, :, 0])
        # action = action + default_dof_pos/0.2
        #-----------------------------------
        obs, _, _, extras = env.step(action)
        # print("obs2", obs)
        # print("env.fld_module.latent_encoding[:, :, 0]2",env.fld_module.latent_encoding[:, :, 0])
        env.render()


    