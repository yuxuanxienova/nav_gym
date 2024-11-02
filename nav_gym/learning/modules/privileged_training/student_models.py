# torch
import torch
import torch.nn as nn
import numpy as np

# rsl-rl
from nav_gym.learning.modules.privileged_training.teacher_models import get_activation
from nav_gym.learning.modules.actor_critic_recurrent import Memory
from nav_gym.learning.modules.submodules.mlp_modules import MLP
from nav_gym.learning.modules.submodules.cnn_modules import SimpleCNN
from nav_gym.learning.modules.normalizer_module import EmpiricalNormalization


class GRUIMUStudentModelRayCast1D(nn.Module):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__()

        self.obs_shape_dict = obs_shape_dict
        self.prop_size = self.obs_shape_dict["prop_student"][0]
        self.exte_size = self.obs_shape_dict["exte_student"][0]
        self.imu_size = self.obs_shape_dict["imu"][0]
        self.num_obs = self.prop_size + self.exte_size + self.imu_size

        self.action_size = action_size

        self.prop_normalizer = EmpiricalNormalization(shape=[self.prop_size], until=1.0e8)
        self.exte_normalizer = EmpiricalNormalization(shape=[self.exte_size], until=1.0e8)
        self.imu_normalizer = EmpiricalNormalization(shape=[self.imu_size], until=1.0e8)

        # Load configs
        self.activation_fn = get_activation(cfg["activation_fn"])

        self.scan_latent_size = cfg["scan_latent_size"]

        self.exte_encoder = MLP(
            cfg["scan_encoder_shape"],
            self.activation_fn,
            self.exte_size,
            self.scan_latent_size,
            init_scale=1.0 / np.sqrt(2),
        )

        if cfg["imu_encoder_shape"] is None:
            self.imu_encoder = None
            self.imu_latent_size = self.imu_size
        else:
            self.imu_encoder = MLP(
                cfg["imu_encoder_shape"],
                self.activation_fn,
                self.priv_size,
                cfg["imu_latent_size"],
                init_scale=1.0 / np.sqrt(2),
            )
            self.imu_latent_size = cfg["imu_latent_size"]

        self.latent_size = self.imu_latent_size + self.scan_latent_size + self.prop_size

        self.memory = Memory(
            self.latent_size, type="gru", num_layers=cfg["gru_num_layers"], hidden_size=cfg["gru_hidden_size"]
        )

        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            cfg["gru_hidden_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )

        self.critic_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            cfg["gru_hidden_size"],
            1,
            init_scale=1.0 / np.sqrt(2),
        )

    def reset(self, dones=None):
        self.memory.reset(dones)

    def get_hidden_states(self):
        return self.memory.hidden_states, self.memory.hidden_states

    def forward(self, obs: torch.Tensor, masks=None, hidden_states=None) -> torch.Tensor:
        observation_latent = self.encode_observation(obs, batch_mode=(masks is not None))
        memory_output = self.memory(observation_latent, masks, hidden_states).squeeze(0)
        output = self.action_head(memory_output)
        return output

    def forward_inference(self, obs: torch.Tensor) -> torch.Tensor:
        observation_latent = self.encode_observation(obs, batch_mode=False)
        memory_output = self.memory(observation_latent).squeeze(0)
        output = self.action_head(memory_output)
        return output

    def forward_critic(self, obs: torch.Tensor, masks=None, hidden_states=None) -> torch.Tensor:
        observation_latent = self.encode_observation(obs, batch_mode=(masks is not None))
        memory_output = self.memory(observation_latent, masks, hidden_states).squeeze(0)
        output = self.critic_head(memory_output)
        return output

    def encode_observation(self, obs: torch.Tensor, batch_mode=True) -> torch.Tensor:
        if batch_mode:
            seq_length = obs.shape[0]
        else:
            seq_length = 1

        obs_flattened = obs.reshape(-1, self.num_obs)

        prop, exte, imu = self.split_obs(obs_flattened)

        prop = self.prop_normalizer(prop)
        exte = self.exte_normalizer(exte)
        imu = self.imu_normalizer(imu)

        # scan
        scan_latent = self.exte_encoder(exte)

        # priv_latent
        if self.imu_encoder is None:
            imu_latent = imu
        else:
            imu_latent = self.imu_encoder(imu)

        concatenated = torch.cat([prop, scan_latent, imu_latent], dim=1)

        if batch_mode:
            num_batches = concatenated.shape[0] // seq_length
            return concatenated.reshape(seq_length, num_batches, self.latent_size)
        else:
            return concatenated

    def split_obs(self, obs):
        return torch.split(obs, [self.prop_size, self.exte_size, self.imu_size], dim=1)


class GRUIMUStudentModelConv2D(nn.Module):
    def __init__(self, obs_shape_dict, action_size, cfg, **kwargs):
        super().__init__()

        self.obs_shape_dict = obs_shape_dict
        self.prop_size = self.obs_shape_dict["prop_student"][0]
        self.imu_size = self.obs_shape_dict["imu"][0]

        self.image_shape = self.obs_shape_dict["exte_student"]

        # self.single_image_shape = list(self.image_shape)
        self.num_images = self.image_shape[0]
        # self.single_image_size = self.single_image_shape[1] * self.single_image_shape[2]
        # self.single_image_shape[0] = 1

        self.exte_size = self.image_shape.numel()

        self.num_obs = self.prop_size + self.exte_size + self.imu_size

        self.action_size = action_size

        self.prop_normalizer = EmpiricalNormalization(shape=[self.prop_size], until=1.0e8)
        # self.exte_normalizer = EmpiricalNormalization(shape=[self.single_image_size], until=1.0e8)
        self.exte_normalizer = EmpiricalNormalization(shape=[self.exte_size], until=1.0e8)
        self.imu_normalizer = EmpiricalNormalization(shape=[self.imu_size], until=1.0e8)

        # Load configs
        self.activation_fn = get_activation(cfg["activation_fn"])

        self.img_latent_size = cfg["img_latent_size"]

        self.exte_encoder = SimpleCNN(
            self.image_shape,
            cfg["img_encoder_channels"],
            cfg["img_encoder_fc_shape"],
            self.img_latent_size,
            self.activation_fn,
            groups=self.num_images,
        )

        if cfg["imu_encoder_shape"] is None:
            self.imu_encoder = None
            self.imu_latent_size = self.imu_size
        else:
            self.imu_encoder = MLP(
                cfg["imu_encoder_shape"],
                self.activation_fn,
                self.priv_size,
                cfg["imu_latent_size"],
                init_scale=1.0 / np.sqrt(2),
            )
            self.imu_latent_size = cfg["imu_latent_size"]

        self.latent_size = self.imu_latent_size + self.img_latent_size + self.prop_size

        self.memory = Memory(
            self.latent_size, type="gru", num_layers=cfg["gru_num_layers"], hidden_size=cfg["gru_hidden_size"]
        )

        self.action_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            cfg["gru_hidden_size"],
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )

        self.critic_head = MLP(
            cfg["output_mlp_size"],
            self.activation_fn,
            cfg["gru_hidden_size"],
            1,
            init_scale=1.0 / np.sqrt(2),
        )

    def reset(self, dones=None):
        self.memory.reset(dones)

    def get_hidden_states(self):
        return self.memory.hidden_states, self.memory.hidden_states

    def forward(self, obs: torch.Tensor, masks=None, hidden_states=None) -> torch.Tensor:
        observation_latent = self.encode_observation(obs, batch_mode=(masks is not None))
        memory_output = self.memory(observation_latent, masks, hidden_states).squeeze(0)
        output = self.action_head(memory_output)
        return output

    def forward_inference(self, obs: torch.Tensor) -> torch.Tensor:
        observation_latent = self.encode_observation(obs, batch_mode=False)
        memory_output = self.memory(observation_latent).squeeze(0)
        output = self.action_head(memory_output)
        return output

    def forward_critic(self, obs: torch.Tensor, masks=None, hidden_states=None) -> torch.Tensor:
        observation_latent = self.encode_observation(obs, batch_mode=(masks is not None))
        memory_output = self.memory(observation_latent, masks, hidden_states).squeeze(0)
        output = self.critic_head(memory_output)
        return output

    def encode_observation(self, obs: torch.Tensor, batch_mode=True) -> torch.Tensor:
        if batch_mode:
            seq_length = obs.shape[0]
        else:
            seq_length = 1

        obs_flattened = obs.reshape(-1, self.num_obs)

        prop, exte, imu = self.split_obs(obs_flattened)

        prop = self.prop_normalizer(prop)
        imu = self.imu_normalizer(imu)

        # scan
        # exte = exte.reshape(-1, self.single_image_size)
        exte = self.exte_normalizer(exte)
        scan_latent = self.exte_encoder(exte)
        # scan_latent = scan_latent.reshape(-1, self.img_latent_size)

        # priv_latent
        if self.imu_encoder is None:
            imu_latent = imu
        else:
            imu_latent = self.imu_encoder(imu)

        concatenated = torch.cat([prop, scan_latent, imu_latent], dim=1)

        if batch_mode:
            num_batches = concatenated.shape[0] // seq_length
            return concatenated.reshape(seq_length, num_batches, self.latent_size)
        else:
            return concatenated

    def split_obs(self, obs):
        return torch.split(obs, [self.prop_size, self.exte_size, self.imu_size], dim=1)
