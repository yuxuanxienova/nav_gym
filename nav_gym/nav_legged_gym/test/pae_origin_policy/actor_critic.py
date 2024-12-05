import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from nav_gym.learning.modules.normalizer import EmpiricalNormalization
from nav_gym.learning.modules.mlp import MLP


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        update_obs_norm=True,
        ext_cfg=None,
        num_prop_obs_first=57,
        num_priv=0,
        num_exte=0,
        learn_std=False,
        force_stop=True,
        latent_dim=4,
        log_offset=-1,
        log_std_min=-20,
        log_std_max=0,
        empirical_normalization=True,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ActorCritic, self).__init__()
        activation = get_activation(activation)

        self.learn_std = learn_std
        self.force_stop = force_stop
        self.empirical_normalization = empirical_normalization
        self.latent_dim = latent_dim
        self.num_priv = num_priv
        if ext_cfg is not None and ext_cfg.use_ext_encoder:
            self.num_ext_obs = ext_cfg.num_ext_obs
            self.num_navi_obs = ext_cfg.num_navi_obs
            self.use_navi_scan_enc = ext_cfg.use_navi_scan_enc
            self.num_prop_obs_first = num_prop_obs_first
            self.num_prop_obs_second = num_actor_obs - num_prop_obs_first - self.num_priv - self.num_ext_obs
            if self.use_navi_scan_enc and self.num_navi_obs != 0:
                self.num_ext_obs_per_channel = (self.num_ext_obs - self.num_navi_obs) // (ext_cfg.ext_num_channels - 1)
            else:
                self.num_ext_obs_per_channel = ext_cfg.num_ext_obs // ext_cfg.ext_num_channels
            self.num_ext_channels = ext_cfg.ext_num_channels
            self.ext_latent_dim = ext_cfg.ext_latent_dim
            ext_hidden_dims = ext_cfg.ext_hidden_dims
            self.ext_encoder = MLP(
                    self.num_ext_obs_per_channel, self.ext_latent_dim, ext_hidden_dims, "elu", last_activation="elu"
                )
            self.ext_navi_encoder = MLP(
                    self.num_navi_obs, self.ext_latent_dim, ext_hidden_dims, "elu", last_activation="elu"
                )
        else:
            self.ext_encoder = None
            self.ext_navi_encoder = None
            self.num_ext_obs = num_exte
            self.num_prop_obs_first = num_prop_obs_first
            self.num_prop_obs_second = num_actor_obs - num_prop_obs_first - self.num_priv - self.num_ext_obs

        mlp_input_dim_a = num_actor_obs if self.ext_encoder is None else num_actor_obs - self.num_ext_obs + self.num_ext_channels * self.ext_latent_dim
        mlp_input_dim_c = num_critic_obs if self.ext_encoder is None else num_actor_obs - self.num_ext_obs + self.num_ext_channels * self.ext_latent_dim

        # Policy
        actor_layers = []
        if self.empirical_normalization:
            actor_layers.append(
                EmpiricalNormalization(shape=[mlp_input_dim_a], update_obs_norm=update_obs_norm, until=1.0e8)
            )
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                if self.learn_std:
                    self.shared_feature = nn.Sequential(*actor_layers) # separate hidden feature and output layer
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        if self.learn_std:
            self.actor = nn.Linear(actor_hidden_dims[layer_index], num_actions) # separate hidden feature and output layer
        else:
            self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        if self.empirical_normalization:
            critic_layers.append(
                EmpiricalNormalization(shape=[mlp_input_dim_c], update_obs_norm=update_obs_norm, until=1.0e8)
            )
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        if self.learn_std:
            self.log_offset = log_offset
            self.log_std_min = log_std_min
            self.log_std_max = log_std_max
            self.latent_actor_logstd = nn.Linear(actor_hidden_dims[layer_index], 3)
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions - 3 * self.latent_dim))
        else:
            self.latent_actor_logstd = None
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        if self.learn_std:
            print(f"Shared MLP: {self.shared_feature}")
            print(f"logstd MLP: {self.latent_actor_logstd}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Exteroception Encoder: {self.ext_encoder}")
        print(f"Velodyne Encoder: {self.ext_navi_encoder}")

        self.distribution = None
    # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]
        
    def init_weights_uniform_(self, scales):
        [
            module.weight.data.uniform_(-scales, scales)
            for idx, module in enumerate(mod for mod in self.actor if isinstance(mod, nn.Linear))
        ]
        [
            module.bias.data.uniform_(-scales, scales)
            for idx, module in enumerate(mod for mod in self.actor if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    @property
    def entropy_partial(self, idx):
        return self.distribution.entropy()[idx, :].sum(dim=-1)

    def update_distribution(self, observations):
        if self.learn_std:
            features = self.shared_feature(observations)
            mean = self.actor(features)
            latent_logstd = self.latent_actor_logstd(features)  
            self.latent_std = F.softplus(torch.clamp(latent_logstd + self.log_offset, self.log_std_min, self.log_std_max)).repeat((1, self.latent_dim))
            std = torch.cat([self.std[: self.force_stop + self.latent_dim].repeat((mean.shape[0], 1)), self.latent_std, self.std[self.force_stop + self.latent_dim :].repeat((mean.shape[0], 1))], dim=-1)
            self.distribution = Normal(mean, mean * 0.0 + std)
        else:
            mean = self.actor(observations)
            self.distribution = Normal(mean, mean * 0.0 + self.std.clip(min=0.05, max=1.0))

    def act(self, observations, **kwargs):
        if self.ext_encoder is not None:
            prop_first, priv, ext, prop_second = self._split_obs(observations)
            if self.use_navi_scan_enc and self.num_navi_obs != 0:
                foot_exte, navi_exte = torch.split(ext, [self.num_ext_obs - self.num_navi_obs, self.num_navi_obs], dim=-1)
                foot_exte = foot_exte.view(
                    -1, self.num_ext_channels - 1, self.num_ext_obs_per_channel
                )
                foot_exte_latent = self.ext_encoder(foot_exte).view(-1, (self.num_ext_channels - 1) * self.ext_latent_dim)
                navi_exte_latent = self.ext_navi_encoder(navi_exte)
                ext_latent = torch.cat((foot_exte_latent, navi_exte_latent), dim=-1)
            else:
                ext = ext.view(
                    -1, self.num_ext_channels, self.num_ext_obs_per_channel
                )
                ext_latent = self.ext_encoder(ext).view(-1, self.num_ext_channels * self.ext_latent_dim)
            observations = torch.cat([prop_first, priv, ext_latent, prop_second], dim=-1)
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_latent_prob(self):
        latent_actions = self.action_mean[:, self.force_stop + self.latent_dim : self.force_stop + self.latent_dim * 4]
        latent_std = self.latent_std[:, :]
        latent_distribution = Normal(latent_actions, latent_actions * 0.0 + latent_std)
        probs = [latent_distribution.log_prob(latent_action).exp().mean(dim=-1) for latent_action in latent_actions]
        self.latent_probs = torch.stack(probs)
        return self.latent_probs
    
    def get_latent_entropy(self, idx=None):
        # latent_probs = self.get_latent_prob(idx)
        if idx is None:
            # if idx is None, we assume that we want to get the prob of all latent actions, idx = [True ,...., True]
            idx = torch.tensor([True] * self.action_mean.shape[0])
        return -torch.sum((self.latent_probs[idx] * torch.log(self.latent_probs[idx])).mean(dim=-1), dim=-1)

    def act_inference(self, observations):
        if self.ext_encoder is not None:
            prop_first, priv, ext, prop_second = self._split_obs(observations)
            if self.use_navi_scan_enc and self.num_navi_obs != 0:
                foot_exte, navi_exte = torch.split(ext, [self.num_ext_obs - self.num_navi_obs, self.num_navi_obs], dim=-1)
                foot_exte = foot_exte.view(
                    -1, self.num_ext_channels - 1, self.num_ext_obs_per_channel
                )
                foot_exte_latent = self.ext_encoder(foot_exte).view(-1, (self.num_ext_channels - 1) * self.ext_latent_dim)
                navi_exte_latent = self.ext_navi_encoder(navi_exte)
                ext_latent = torch.cat((foot_exte_latent, navi_exte_latent), dim=-1)
            else:
                ext = ext.view(
                    -1, self.num_ext_channels, self.num_ext_obs_per_channel
                )
                ext_latent = self.ext_encoder(ext).view(-1, self.num_ext_channels * self.ext_latent_dim)
            observations = torch.cat([prop_first, priv, ext_latent, prop_second], dim=-1)
        if self.learn_std:
            hidden_features = self.shared_feature(observations)
            actions_mean = self.actor(hidden_features)
        else:
            actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        if self.ext_encoder is not None:
            prop_first, priv, ext, prop_second = self._split_obs(critic_observations)
            if self.use_navi_scan_enc and self.num_navi_obs != 0:
                foot_exte, navi_exte = torch.split(ext, [self.num_ext_obs - self.num_navi_obs, self.num_navi_obs], dim=-1)
                foot_exte = foot_exte.view(
                    -1, self.num_ext_channels - 1, self.num_ext_obs_per_channel
                )
                foot_exte_latent = self.ext_encoder(foot_exte).view(-1, (self.num_ext_channels - 1) * self.ext_latent_dim)
                navi_exte_latent = self.ext_navi_encoder(navi_exte)
                ext_latent = torch.cat((foot_exte_latent, navi_exte_latent), dim=-1)
            else:
                ext = ext.view(
                    -1, self.num_ext_channels, self.num_ext_obs_per_channel
                )
                ext_latent = self.ext_encoder(ext).view(-1, self.num_ext_channels * self.ext_latent_dim)
            critic_observations = torch.cat([prop_first, priv, ext_latent, prop_second], dim=-1)
        value = self.critic(critic_observations)
        return value
    
    def _split_obs(self, observations: torch.Tensor):
        """Split the observations into prop and ext components

        Args:
            observations (torch.Tensor): tensor of observations

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: prop and ext observations
        """
        prop_first, priv, ext, prop_second = torch.split(observations, [self.num_prop_obs_first, self.num_priv, self.num_ext_obs, self.num_prop_obs_second], dim=-1)
        return prop_first, priv, ext, prop_second
    
    def split_obs(self, observations: torch.Tensor):
        """Same as _split_obs but public method which returns a dictionary, to avoid return signature issues"""
        prop_first, priv, ext, prop_second = self._split_obs(observations)
        return {"prop": (prop_first, prop_second), "ext": ext, "priv": priv}


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None