#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from nav_gym.learning.modules.actor_critic import ActorCriticSeparate
from nav_gym.learning.storage.rollout_storage import RolloutStorageASE as RolloutStorage
from nav_gym.learning.modules.ase.discriminator_encoder import DiscriminatorEncoder
from nav_gym.learning.modules.ase.amp_demo_storage import AMPDemoStorage
from nav_gym.learning.modules.ase.amp_obs_storage import AMPObsStorage
import math


class PPO_ASE:
    actor_critic: ActorCriticSeparate

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
        num_envs=1,
        num_transitions_per_env=1,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer_actor_critic = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
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

        # ASE parameters
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        
        self.history_length = 2
        self.amp_obs_dim = 30
        self.ase_latent_dim = 32

        self.discriminator_encoder = DiscriminatorEncoder(self.history_length, self.amp_obs_dim, latent_dim=self.ase_latent_dim).to(self.device)
        self.optimizer_dis_enc = optim.Adam(self.discriminator_encoder.parameters(), lr=learning_rate)
        self.amp_demo_storage = AMPDemoStorage()
        self.amp_obs_storage = AMPObsStorage(self.amp_obs_dim, self.num_envs, self.num_transitions_per_env, self.num_transitions_per_env * 2)
        self.num_samples = self.num_envs

        
        self.ase_latents = torch.zeros((self.num_envs, self.ase_latent_dim), dtype=torch.float32,device=self.device)

        self.scale_disc_grad_penalty = 10.0
        self._dibsc_logit_reg = 0.01
        self._disc_weight_decay = 0.0001
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
        action, _ = self.actor_critic.act(obs)
        #action: (num_envs, action_shape)
        return action
    def get_log_prob(self, action):
        log_prob=self.actor_critic.get_actions_log_prob(action)
        return log_prob
    def store_transition(self, action, log_prob, obs, critic_obs,log_disc_prob):
        #log_prob: (num_envs, 1)
        self.transition.actions = action.detach()
        self.transition.actions_log_prob = log_prob.detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        #ASE
        self.transition.log_disc_prob = log_disc_prob.detach()
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

    def update(self):
        self.num_updates += 1
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_bonus = 0
        mean_disc_loss = 0
        mean_enc_loss = 0
        mean_disc_agent_acc = 0
        mean_disc_demo_acc = 0

        self.entropy_coef = self.adjust_coeff_exp(self.entropy_coef_init, self.num_updates, self.entropy_coef_decay)
        # print("[DEBUG] entropy_coef", self.entropy_coef)

        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            log_disc_probs_batch,#ASE
            hid_states_batch,
            masks_batch,
            
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            actions_log_prob_batch=torch.clamp(actions_log_prob_batch, min=-100, max=100)

            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            entropy_batch = self.actor_critic.entropy
            #------------ASE-------------
            disc_loss = torch.zeros(1).to(self.device)
            enc_loss = torch.zeros(1).to(self.device)
            disc_agent_acc = torch.zeros(1).to(self.device)
            disc_demo_acc = torch.zeros(1).to(self.device)
            if self.amp_obs_storage.is_ready(self.history_length):
                obs_amp = self.amp_obs_storage.get_current_obs(self.history_length).to(self.device)
                # obs_amp: [num_envs, history_length, obs_dim]
                # obs_amp_replay = self.amp_obs_storage.sample_amp_obs_batch(self.history_length, self.num_samples).to(self.device)
                # obs_amp_replay: [num_samples*num_envs, history_length, obs_dim]
                obs_demo = self.amp_demo_storage.sample_amp_demo_obs_batch(self.history_length, self.num_samples).to(self.device)
                obs_demo.requires_grad = True
                # obs_demo: [num_samples, history_length, obs_dim]
                disc_agent_logit, enc_pred = self.discriminator_encoder(obs_amp)
                # disc_agent_logit: [num_envs, 1]
                # enc_pred: [num_envs, latent_dim]
                # disc_agent_replay_logit,_ = self.discriminator_encoder(obs_amp_replay)
                # disc_agent_replay_logit: [num_samples*num_envs, 1]
                disc_demo_logit,_ = self.discriminator_encoder(obs_demo) 
                # disc_demo_logit: [num_samples, 1]
                # disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
                # disc_agent_cat_logit: [(num_samples+1)*num_envs, 1]
                # disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, obs_demo)
                disc_info = self._disc_loss(disc_agent_logit, disc_demo_logit, obs_demo)
                disc_loss = disc_info['disc_loss']

                #other info
                disc_grad_penalty = disc_info['disc_grad_penalty']
                disc_logit_loss = disc_info['disc_logit_loss']
                disc_agent_acc = disc_info['disc_agent_acc']
                disc_demo_acc = disc_info['disc_demo_acc']
                

                enc_latents = self.ase_latents
                # enc_latents: [num_envs, latent_dim]

                # rand_action_mask = torch.bernoulli(rand_action_probs)
                # enc_loss_mask = rand_action_mask[0:self._amp_minibatch_size]
                # enc_info = self._enc_loss(enc_pred, enc_latents, batch_dict['amp_obs'], enc_loss_mask)

                enc_info = self._enc_loss(enc_pred, enc_latents)
                enc_loss = enc_info['enc_loss']

                #Discriminator Encoder Loss
                # loss_dis_enc = disc_loss + enc_loss
                loss_dis_enc = disc_loss 

                #Gradient Step
                self.optimizer_dis_enc.zero_grad()
                loss_dis_enc.backward()
                nn.utils.clip_grad_norm_(self.discriminator_encoder.parameters(), self.max_grad_norm)
                self.optimizer_dis_enc.step()

            #----------------------------------

            #-------------------PPO-------------------
            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    log_rat = actions_log_prob_batch - old_actions_log_prob_batch.reshape(-1)
                    kl_mean = torch.mean((torch.exp(log_rat) - 1) - log_rat)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-6, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                    for param_group in self.optimizer_actor_critic.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio

            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            #------ase surrogate loss----
            # ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            # surrogate_ase = -torch.squeeze(log_disc_probs_batch) * ratio
            # surrogate_clipped_ase = -torch.squeeze(log_disc_probs_batch) * torch.clamp(
            #     ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            # )
            # surrogate_loss_ase = torch.max(surrogate_ase, surrogate_clipped_ase).mean()

            #----------------------------

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

            # Actor Critic Loss
            loss_actor_critic = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_bonus 

            # Gradient step
            self.optimizer_actor_critic.zero_grad()
            loss_actor_critic.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer_actor_critic.step()

            #-----------------------------------


            #Calculate the mean loss
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_bonus += entropy_bonus.item()
            mean_disc_loss += disc_loss.item()
            mean_enc_loss += enc_loss.item()
            mean_disc_agent_acc += disc_agent_acc.item()
            mean_disc_demo_acc += disc_demo_acc.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_bonus /= num_updates
        mean_disc_loss /= num_updates
        mean_enc_loss /= num_updates
        mean_disc_agent_acc /= num_updates
        mean_disc_demo_acc /= num_updates   
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy_bonus, mean_disc_loss, mean_enc_loss, mean_disc_agent_acc, mean_disc_demo_acc

    def adjust_coeff_exp(self, initial_value, iteration, decay_rate):
        r = initial_value * math.pow(decay_rate, iteration)
        if r < 1.0e-7:
            r = 1.0e-7
        return r
    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # disc_agent_logit: [(num_samples+1)*num_envs, 1]
        # disc_demo_logit: [num_samples, 1]
        # obs_demo: [num_samples, history_length, obs_dim]
        # prediction loss
        #--------------------loss1----------------------
        # disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        # disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        # disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)
        #--------------------loss2---------------------
        disc_loss_agent = torch.sum(torch.square(disc_agent_logit + 1.0))
        disc_loss_demo = torch.sum(torch.square(disc_demo_logit - 1.0))
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.discriminator_encoder.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss = disc_loss + self._dibsc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss = disc_loss + self.scale_disc_grad_penalty * disc_grad_penalty

        # weight decay
        disc_weights = self.discriminator_encoder.get_disc_weights()
        disc_weight_decay = torch.sum(torch.square(disc_weights))
        disc_loss = disc_loss + self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty.detach(),
            'disc_logit_loss': disc_logit_loss.detach(),
            'disc_agent_acc': disc_agent_acc.detach(),
            'disc_demo_acc': disc_demo_acc.detach(),
            'disc_agent_logit': disc_agent_logit.detach(),
            'disc_demo_logit': disc_demo_logit.detach()
        }
        return disc_info 
    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss
    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss
    def _enc_loss(self, enc_pred, ase_latent, enc_obs=None, loss_mask=None):
        # enc_pred: [num_envs, latent_dim]
        enc_err = self._calc_enc_error(enc_pred, ase_latent)
        #mask_sum = torch.sum(loss_mask)
        #enc_err = enc_err.squeeze(-1)
        #enc_loss = torch.sum(loss_mask * enc_err) / mask_sum
        enc_loss = torch.mean(enc_err)

        #encoder weight decay
        # if (self._enc_weight_decay != 0):
        #     enc_weights = self.model.a2c_network.get_enc_weights()
        #     enc_weights = torch.cat(enc_weights, dim=-1)
        #     enc_weight_decay = torch.sum(torch.square(enc_weights))
        #     enc_loss += self._enc_weight_decay * enc_weight_decay
            
        enc_info = {
            'enc_loss': enc_loss
        }

        #encoder grad penalty
        # if (self._enable_enc_grad_penalty()):
        #     enc_obs_grad = torch.autograd.grad(enc_err, enc_obs, grad_outputs=torch.ones_like(enc_err),
        #                                        create_graph=True, retain_graph=True, only_inputs=True)
        #     enc_obs_grad = enc_obs_grad[0]
        #     enc_obs_grad = torch.sum(torch.square(enc_obs_grad), dim=-1)
        #     #enc_grad_penalty = torch.sum(loss_mask * enc_obs_grad) / mask_sum
        #     enc_grad_penalty = torch.mean(enc_obs_grad)

        #     enc_loss += self._enc_grad_penalty * enc_grad_penalty

        #     enc_info['enc_grad_penalty'] = enc_grad_penalty.detach()

        return enc_info   
    def _calc_enc_error(self, enc_pred, ase_latent):
        err = enc_pred * ase_latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        """
        Computes the discriminator accuracy for agent and demonstration logits.

        Args:
            disc_agent_logit (torch.Tensor): Logits for agent samples, shape [N_agent, 1]
            disc_demo_logit (torch.Tensor): Logits for demo samples, shape [N_demo, 1]

        Returns:
            tuple: (disc_agent_acc, disc_demo_acc)
                - disc_agent_acc (torch.Tensor): Accuracy for agent samples (scalar tensor)
                - disc_demo_acc (torch.Tensor): Accuracy for demo samples (scalar tensor)
        """
        # Ensure logits are of shape [batch_size, 1]
        assert disc_agent_logit.dim() == 2 and disc_agent_logit.size(1) == 1, \
            "disc_agent_logit should have shape [N_agent, 1]"
        assert disc_demo_logit.dim() == 2 and disc_demo_logit.size(1) == 1, \
            "disc_demo_logit should have shape [N_demo, 1]"

        # Apply sigmoid to convert logits to probabilities
        agent_probs = torch.sigmoid(disc_agent_logit)  # Shape: [N_agent, 1]
        demo_probs = torch.sigmoid(disc_demo_logit)    # Shape: [N_demo, 1]

        # Predicted labels based on probability threshold of 0.5
        agent_preds = (agent_probs >= 0.5).float()  # Shape: [N_agent, 1]
        demo_preds = (demo_probs >= 0.5).float()    # Shape: [N_demo, 1]

        # True labels: 0 for agent, 1 for demo
        agent_labels = torch.zeros_like(agent_preds)  # Shape: [N_agent, 1]
        demo_labels = torch.ones_like(demo_preds)     # Shape: [N_demo, 1]

        # Calculate correct predictions
        agent_correct = (agent_preds == agent_labels).float()  # Shape: [N_agent, 1]
        demo_correct = (demo_preds == demo_labels).float()     # Shape: [N_demo, 1]

        # Compute mean accuracy
        disc_agent_acc = agent_correct.mean()  # Scalar tensor
        disc_demo_acc = demo_correct.mean()    # Scalar tensor

        return disc_agent_acc, disc_demo_acc
    def sample_latents(self, n):
        z = torch.normal(torch.zeros([n, self.ase_latent_dim], device=self.device))
        z = torch.nn.functional.normalize(z, dim=-1)
        return z
    
    #--------------getters and setters----------------
    def get_ase_latents(self):
        return self.ase_latents
    def set_ase_latents(self, z, new_latent_env_ids):
        self.ase_latents[new_latent_env_ids] = z
