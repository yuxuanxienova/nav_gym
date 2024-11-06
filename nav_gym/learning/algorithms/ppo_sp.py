#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from nav_gym.learning.modules.actor_critic import ActorCritic
from nav_gym.learning.storage.rollout_storage import RolloutStorage

import math
from nav_gym.learning.storage.rollout_storage import TrajectoryStorage

class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        entropy_coef_decay=0.999,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.entropy_coef_init = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.num_updates = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        #obs: (num_envs, actor_obs_shape)
        action, log_prob = self.actor_critic.act(obs)
        #action: (num_envs, action_shape)
        #log_prob: (num_envs, 1)
        self.transition.actions = action.detach()
        self.transition.actions_log_prob = log_prob.detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, trajectory_storage: TrajectoryStorage):
        self.num_updates += 1
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_bonus = 0

        self.entropy_coef = self.adjust_coeff_exp(self.entropy_coef_init, self.num_updates, self.entropy_coef_decay)

        generator = trajectory_storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # Validate input tensors
            input_tensors = {
                'obs_batch': obs_batch,
                'critic_obs_batch': critic_obs_batch,
                'actions_batch': actions_batch,
                'target_values_batch': target_values_batch,
                'advantages_batch': advantages_batch,
                'returns_batch': returns_batch,
                'old_actions_log_prob_batch': old_actions_log_prob_batch,
            }

            samples, log_prob = self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            eps = 1e-6
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch + eps)

            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )

            entropy_batch = self.actor_critic.entropy

            # Calculate ratio and clamp it
            diff = actions_log_prob_batch - old_actions_log_prob_batch.squeeze()
            diff = torch.clamp(diff, min=-10, max=10)  # Adjust the range as appropriate
            ratio = torch.exp(diff)
            # ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze())#problem in actions_log_prob_batch
            ratio = torch.clamp(ratio, 0.0, 10.0)

            # Surrogate loss
            surrogate = -advantages_batch.squeeze() * ratio#problem in ratio !!
            surrogate_clipped = -advantages_batch.squeeze() * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Entropy bonus
            entropy_bonus = entropy_batch.mean()

            # Total loss
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_bonus#problem in surrogateloss@!!

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            print("loss",loss)

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_bonus += entropy_bonus.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_bonus /= num_updates
        # self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy_bonus

    def adjust_coeff_exp(self, initial_value, iteration, decay_rate):
        r = initial_value * math.pow(decay_rate, iteration)
        if r < 1.0e-7:
            r = 1.0e-7
        return r
