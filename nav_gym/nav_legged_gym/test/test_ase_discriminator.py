
from nav_gym.learning.modules.ase.amp_demo_storage import AMPDemoStorage
from nav_gym.learning.modules.ase.discriminator_encoder import DiscriminatorEncoder
import os
from nav_gym import NAV_GYM_ROOT_DIR

import torch
import torch.optim as optim

if __name__ == "__main__":
    # ----------------------------1. Load the DiscriminatorEncoder model----------------------------
    # Define model parameters
    history_length = 16
    num_dof = 24
    latent_dim = 32

    # Set device to CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize a new model instance with the same parameters
    model = DiscriminatorEncoder(history_length, num_dof, latent_dim).to(device)

    # Initialize the optimizer (must match the original optimizer's type and parameters)
    loaded_optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Load the saved states
    log_dir = os.path.join(os.path.dirname(__file__), "logs/locomotion_ase/20241119-160833/")
    save_path = os.path.join(log_dir, "discriminator_encoder_300.pt")
    model.load(save_path, map_location=device, optimizer=loaded_optimizer)

    # Set the model to evaluation mode
    model.eval()

    # ----------------------------2. Load the Recorded Motion Data----------------------------
    datasets_root = os.path.join(NAV_GYM_ROOT_DIR, "resources/anymal_d/datasets/record/")
    motion_names = "sampled_amp_observations.pt"
    save_path = os.path.join(datasets_root, motion_names)
    loaded_amp_obs = torch.load(save_path, map_location=device)
    print(f"Loaded AMP observations from {save_path}")

    state_index_dict = {
        "dof_pos": [0, 12],
        "dof_vel": [12, 24]
    }

    amp_record_obs = loaded_amp_obs[0].to(device)  # Shape: [sample_length, amp_obs_dim]

    # -----------------------------3. Load the Demonstration Motion Data----------------------------
    # Load the Motion Data
    amp_demo_storage = AMPDemoStorage()
    sample_length = 16
    amp_demo_obs = amp_demo_storage.sample_amp_demo_obs_single(sample_length).to(device)  # Shape: [sample_length, amp_obs_dim]

    # -----------------------------4. Compute the Discriminator Loss----------------------------
    with torch.no_grad():
        # Recorded Motion Data
        # Assuming the model expects input shape: (batch_size, history_length, num_dof)
        # If amp_record_obs is (history_length, num_dof), add batch dimension
        amp_record_obs_batch = amp_record_obs.unsqueeze(0)  # Shape: [1, history_length, num_dof]
        disc_logits, mu_q = model(amp_record_obs_batch)
        prob = torch.sigmoid(disc_logits)
        print("----------Recorded Motion Data-----------")
        print(f"Discriminator Output: {disc_logits}")
        print(f"Discriminator Probabilities: {prob}")

        # Demonstration Motion Data
        amp_demo_obs_batch = amp_demo_obs.unsqueeze(0)  # Shape: [1, history_length, num_dof]
        disc_logits, mu_q = model(amp_demo_obs_batch)
        prob = torch.sigmoid(disc_logits)
        print("----------Demonstration Motion Data-----------")
        print(f"Discriminator Output: {disc_logits}")
        print(f"Discriminator Probabilities: {prob}")
