from nav_gym.nav_legged_gym.envs.config_locomotion_env import LocomotionEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
import torch
import os
import time
if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "logs/" + time.strftime("%Y%m%d-%H%M%S"))
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "logs/20241025-233550/" + "model_300.pt")
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)


    env = LocomotionEnv(LocomotionEnvCfg())
    runner = OnPolicyRunner(env,train_cfg_dict , log_dir=log_dir, device="cuda:0")
    runner.load(checkpoint_dir)
    policy = runner.get_inference_policy()
    obs, extras = env.reset()
    while True:
        action = policy(obs)
        obs, _,_, _, extras = env.step(action)
        env.render()
    