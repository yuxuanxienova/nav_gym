from nav_gym.nav_legged_gym.envs.config_locomotion_env import LocomotionEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.nav_legged_gym.utils.helpers import export_policy_as_jit
import torch
import os
import time
from nav_gym import NAV_GYM_ROOT_DIR

if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/" + time.strftime("%Y%m%d-%H%M%S"))
    checkpoint_dir = os.path.join(os.path.dirname(NAV_GYM_ROOT_DIR), "logs/20241103-092555/" + "model_26099.pt")
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)
    
    # Initialize the environment
    env = LocomotionEnv(LocomotionEnvCfg())
    
    # Initialize the runner
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    
    # Load the checkpoint
    runner.load(checkpoint_dir)
    # runner.alg.actor_critic.actor[0]._is_training = False
    policy = runner.get_inference_policy(device=env.device)

    
    # Get the policy model
    # policy = runner.get_inference_policy()
    
    # Prepare an example input for tracing
    # Assuming the policy model expects observations as input
    # Let's get an observation from the environment
    
    # Reset the environment to get initial observations
    obs, extras = env.reset()
    
    # Check if obs is already a tensor; if not, convert it to a tensor
    if not isinstance(obs, torch.Tensor):
        # Convert obs to a torch tensor
        obs = torch.tensor(obs, dtype=torch.float32)
    
    # Add batch dimension if necessary
    if len(obs.shape) == 1:
        # Add batch dimension
        obs = obs.unsqueeze(0)
    
    # Script the policy model
    try:
        scripted_model_path = os.path.join(NAV_GYM_ROOT_DIR, "resources/model/low_level/" )
        file_name = "ll_jit_model2.pt"
        export_policy_as_jit(runner.alg.actor_critic, None, scripted_model_path, filename=file_name)
        print(f"Scripted model saved to {scripted_model_path}")
    except Exception as e:
        print("Scripting failed:", e)
        exit(1)
    
    # Now, let's load the model and test it
    try:
        loaded_policy = torch.jit.load(scripted_model_path + file_name).to("cuda:0")
        loaded_policy.eval()
    except Exception as e:
        print("Loading scripted model failed:", e)
        exit(1)
    
    # Get another observation from the environment
    obs, extras = env.reset()
    
    while True:
        action = loaded_policy(obs)
        obs, _,_, extras = env.step(action)
        env.render()
