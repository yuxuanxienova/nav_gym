
import os 
from nav_gym import NAV_GYM_ROOT_DIR
import random
import torch
class AMPObsStorage:
    def __init__(self,obs_dim,num_envs,num_transitons_per_env, trajectory_capacity):
        #parse args
        self.num_envs = num_envs
        self.capacity = trajectory_capacity
        self.obs_dim = obs_dim
        self.num_transitons_per_env = num_transitons_per_env

        #initialize buffers
        self.transition_buffer = torch.zeros(self.num_envs,self.num_transitons_per_env,self.obs_dim)
        self.data = torch.zeros(self.capacity,self.num_envs,self.num_transitons_per_env,self.obs_dim)
        self.count = 0
        self.length = 0

    def add_amp_obs_to_buffer(self, amp_obs,transition_idx):
        #amp_obs: [num_envs,obs_dim]
        self.transition_buffer[:,transition_idx,:] = amp_obs

    def add_transition_to_data(self):
        self.data[self.count] = self.transition_buffer
        self.count = (self.count + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)
        print(f"[INFO]Count: {self.count}")

    def is_ready(self,sample_length):
        return self.length > sample_length
    def clear_buffer(self):
        self.transition_buffer = torch.zeros(self.num_envs,self.num_transitons_per_env,self.obs_dim)
    # def is_ready(self):
    #     return 
    def get_amp_traj_shape(self):
        return self.data[0].shape
    def get_current_obs(self,sample_length):
        #----Error Handling----
        if self.num_transitons_per_env < sample_length:
            print("[ERROR]Sample length is greater than the number of transitions per environment")
            exit(1)
        #--------------------
        #self.data[self.count]: [num_envs,num_transitons_per_env,obs_dim]
        obs_amp = self.data[self.count-1][:,:sample_length,:]
        #obs_amp: [num_envs,sample_length,obs_dim]
        return obs_amp
    def sample_amp_obs_batch(self, sample_length, num_sample=1):
        #----Error Handling----
        if self.num_transitons_per_env < sample_length:
            print("[ERROR]Sample length is greater than the number of transitions per environment")
            exit(1)
        #--------------------    
        indices = torch.randint(0, self.length, (num_sample,))
        #self.data[indices]: [num_sample,num_envs,num_transitons_per_env,obs_dim]
        obs_amp_replay = self.data[indices][:,:,:sample_length,:].flatten(0,1)  
        #obs_amp_replay: [num_sample*num_envs,sample_length,obs_dim]
        return obs_amp_replay

