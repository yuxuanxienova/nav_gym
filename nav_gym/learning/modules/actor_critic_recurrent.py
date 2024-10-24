#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn

# learning
from nav_gym.learning.modules.actor_critic import ActorCritic, get_activation
from nav_gym.learning.utils import unpad_trajectories
from nav_gym.learning.modules.mlp import MLP


class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        ext_cfg=None,
        num_prop_obs_first=57,
        num_priv=0,
        num_exte=0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_size,
            num_critic_obs=rnn_hidden_size,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            # ext_cfg=ext_cfg,
            num_prop_obs_first=num_prop_obs_first,
            num_priv=num_priv,
            num_exte=num_exte,
        )
        
        self.num_priv = num_priv
        if ext_cfg is not None and ext_cfg.use_ext_encoder:
            self.num_ext_obs = ext_cfg.num_ext_obs
            self.num_prop_obs_first = num_prop_obs_first
            self.num_prop_obs_second = num_actor_obs - num_prop_obs_first - self.num_priv - self.num_ext_obs
            self.num_ext_obs_per_channel = ext_cfg.num_ext_obs // ext_cfg.ext_num_channels
            self.num_ext_channels = ext_cfg.ext_num_channels
            self.ext_latent_dim = ext_cfg.ext_latent_dim
            ext_hidden_dims = ext_cfg.ext_hidden_dims
            self.ext_encoder = MLP(
                    self.num_ext_obs_per_channel, self.ext_latent_dim, ext_hidden_dims, "elu", last_activation="elu"
                )
        else:
            self.ext_encoder = None
            self.num_prop_obs_first = num_prop_obs_first
            self.num_ext_obs = num_exte
            self.num_prop_obs_second = num_actor_obs - num_prop_obs_first - self.num_priv - self.num_ext_obs

        activation = get_activation(activation)
        rnn_input_dim_a = num_actor_obs if self.ext_encoder is None else num_actor_obs - self.num_ext_obs + self.num_ext_channels * self.ext_latent_dim
        rnn_input_dim_c = num_critic_obs if self.ext_encoder is None else num_actor_obs - self.num_ext_obs + self.num_ext_channels * self.ext_latent_dim
        
        self.memory_a = Memory(rnn_input_dim_a, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(rnn_input_dim_c, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
        
        # Policy
        actor_layers = []
        # actor_layers.append(
        #     EmpiricalNormalization(shape=[mlp_input_dim_a], update_obs_norm=update_obs_norm, until=1.0e8)
        # )
        actor_layers.append(nn.Linear(rnn_hidden_size, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        # critic_layers.append(
        #     EmpiricalNormalization(shape=[mlp_input_dim_c], update_obs_norm=update_obs_norm, until=1.0e8)
        # )
        critic_layers.append(nn.Linear(rnn_hidden_size, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    # def act(self, observations, masks=None, hidden_states=None):
    #     input_a = self.memory_a(observations, masks, hidden_states)
    #     return super().act(input_a.squeeze(0))

    # def act_inference(self, observations):
    #     input_a = self.memory_a(observations)
    #     return super().act_inference(input_a.squeeze(0))

    # def evaluate(self, critic_observations, masks=None, hidden_states=None):
    #     input_c = self.memory_c(critic_observations, masks, hidden_states)
    #     return super().evaluate(input_c.squeeze(0))

    def act(self, observations, masks=None, hidden_states=None):
        if self.ext_encoder is not None:
            prop_first, priv, ext, prop_second = self._split_obs(observations)
            seq_length = ext.shape[0]
            ext = ext.view(
                seq_length, -1, self.num_ext_channels, self.num_ext_obs_per_channel
            )
            ext_latent = self.ext_encoder(ext).view(seq_length, -1, self.num_ext_channels * self.ext_latent_dim).squeeze()
            observations = torch.cat([prop_first, priv, ext_latent, prop_second], dim=-1)
        input_a = self.memory_a(observations, masks, hidden_states)
        self.update_distribution(input_a.squeeze(0))
        return self.distribution.sample()

    def act_inference(self, observations):
        if self.ext_encoder is not None:
            prop_first, priv, ext, prop_second = self._split_obs(observations)
            ext = ext.view(
                -1, self.num_ext_channels, self.num_ext_obs_per_channel
            )
            ext_latent = self.ext_encoder(ext).view(-1, self.num_ext_channels * self.ext_latent_dim)
            observations = torch.cat([prop_first, priv, ext_latent, prop_second], dim=-1)
        input_a = self.memory_a(observations)
        actions_mean = self.actor(input_a.squeeze(0))
        return actions_mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        if self.ext_encoder is not None:
            prop_first, priv, ext, prop_second = self._split_obs(critic_observations)
            seq_length = ext.shape[0]
            ext = ext.view(
                seq_length, -1, self.num_ext_channels, self.num_ext_obs_per_channel
            )
            ext_latent = self.ext_encoder(ext).view(seq_length, -1, self.num_ext_channels * self.ext_latent_dim).squeeze()
            observations = torch.cat([prop_first, priv, ext_latent, prop_second], dim=-1)
        input_c = self.memory_c(observations, masks, hidden_states)
        value = self.critic(input_c.squeeze(0))
        return value

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
