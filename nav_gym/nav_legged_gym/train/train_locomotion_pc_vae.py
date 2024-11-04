
from nav_gym.nav_legged_gym.envs.config_locomotion_pc_env import LocomotionEnvCfg
from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.config_train_locomotion import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.learning.modules.VAE import VAE
import torch
import os
import time
import torch.optim as optim  # Import the optimizer
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "logs/" + time.strftime("%Y%m%d-%H%M%S"))
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "logs/20241029-120802/" + "model_4200.pt")
    # log_dir = None
    train_cfg = TrainConfig
    train_cfg_dict = class_to_dict(train_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "pc_encoder"))

    env = LocomotionEnv(LocomotionEnvCfg())
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    runner.load(checkpoint_dir)
    policy = runner.get_inference_policy()
    obs, extras = env.reset()

    # -------------Model--------------
    dim_pc = env.obs_manager.get_obs_dims_from_group("point_cloud")
    save_every = 1000
    input_dim = dim_pc
    latent_dim = 64
    hidden_dim = 256
    batch_size = 16
    vae = VAE(input_dim, latent_dim, hidden_dim)
    vae.to(device)

    # Define the optimizer
    learning_rate = 1e-4  # Adjust the learning rate if necessary
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate,weight_decay=1e-5)

    # Initialize a step counter
    global_step = 0
    while True:
        obs_pc_list = []
        while len(obs_pc_list) < batch_size:
            action = policy(obs)
            obs, _, _, _, extras = env.step(action)
            env.render()

            # ------------Train Point Cloud Encoder------------
            obs_pc = env.obs_manager.get_obs_from_group("point_cloud")

            dim_pc = env.obs_manager.get_obs_dims_from_group("point_cloud")
            obs_pc = obs_pc.view(-1, dim_pc).to(device)  # Ensure the data is on the correct device
            if torch.isnan(obs_pc).any():
                print("Warning: obs contains NaN values!")
                continue
            obs_pc_list.append(obs_pc)

        # Stack observations into a batch
        obs_pc_batch = torch.stack(obs_pc_list, dim=0).to(device)  # Shape: [batch_size, input_dim]
        # Forward pass through the VAE
        recon_pc_batch, mu, log_var = vae(obs_pc_batch)
        loss = vae.loss_function(recon_pc_batch, obs_pc_batch, mu, log_var)

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm)

        optimizer.step()

        # Monitoring
        print(f" Loss: {loss.item()}")

        # Log the loss to TensorBoard
        writer.add_scalar('Loss/Total', loss.item(), global_step)

        # (Optional) Log individual loss components if available
        # If you modify the loss_function to return MSE and KLD separately,
        # you can log them as follows:
        # MSE, KLD = ...  # Extract these from your loss function
        # writer.add_scalar('Loss/MSE', MSE.item(), global_step)
        # writer.add_scalar('Loss/KLD', KLD.item(), global_step)

        # (Optional) Log histograms of model parameters
        # for name, param in vae.named_parameters():
        #     writer.add_histogram(name, param, global_step)

        # (Optional) Log the model graph (only once)
        # if global_step == 0:
        #     writer.add_graph(vae, obs_pc_batch)
                    # Save checkpoint periodically
        if global_step % save_every == 0 and global_step != 0:
            checkpoint_path = os.path.join(log_dir, f'pc_encoder/vae_step_{global_step}.pt')
            vae.save(checkpoint_path, optimizer)
            print(f"Model checkpoint saved at step {global_step} to {checkpoint_path}")


        global_step += 1



