from nav_gym.nav_legged_gym.envs.legged_nav_env_pc_config import LeggedNavEnvCfg
from nav_gym.nav_legged_gym.envs.legged_nav_env import LeggedNavEnv
from nav_gym.learning.runners.on_policy_runner import OnPolicyRunner
from nav_gym.nav_legged_gym.train.train_config import TrainConfig
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
from nav_gym.learning.modules.VAE import VAE,tVAE
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

    env = LeggedNavEnv(LeggedNavEnvCfg())
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    runner.load(checkpoint_dir)
    policy = runner.get_inference_policy()
    obs, extras = env.reset()

    # -------------Model--------------
    dim_pc = env.obs_manager.get_obs_dims_from_group("point_cloud")
    input_dim = dim_pc
    save_every = 500
    latent_dim = 32
    hidden_dim = 128
    batch_size = 16
    num_tokens = 8
    tvae = tVAE(input_dim=input_dim,latent_dim=latent_dim,hidden_dim= hidden_dim,num_input_tokens= num_tokens)
    tvae.to(device)

    # Define the optimizer
    learning_rate = 1e-3  # Adjust the learning rate if necessary
    optimizer = optim.Adam(tvae.parameters(), lr=learning_rate)

    # Initialize a step counter
    global_step = 0
    while True:
        obs_pc_tokens_list = []
        while len(obs_pc_tokens_list) < num_tokens:
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
            obs_pc_tokens_list.append(obs_pc_batch)

        obs_pc_tokens_batch = torch.stack(obs_pc_tokens_list, dim=1).squeeze().to(device)  # Shape: [batch_size, num_tokens,  input_dim]
        # Forward pass through the VAE
        recon_pc_tokens_batch, mu, log_var = tvae(obs_pc_tokens_batch)
        loss = tvae.loss_function(recon_pc_tokens_batch, obs_pc_tokens_batch, mu, log_var)

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(tvae.parameters(), max_norm)

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
        if global_step % save_every == 0 and global_step != 0:
            checkpoint_path = os.path.join(log_dir, f'pc_encoder/vae_step_{global_step}.pt')
            tvae.save(checkpoint_path, optimizer)
            print(f"Model checkpoint saved at step {global_step} to {checkpoint_path}")

        global_step += 1