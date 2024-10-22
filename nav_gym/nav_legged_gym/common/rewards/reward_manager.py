import torch


class RewardManager:
    def __init__(self, env):
        """Prepares a list of reward functions, which will be called to compute the total reward.
        Looks for self.<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        #1. Initializing Attributes
        self.reward_functions = {}
        self.reward_params = {}
        self.episode_sums = {}
        self.reward = 0.0
        self.only_positive_rewards = False
        if env.cfg.__dict__.get("rewards", None) is None:
            return
        self.only_positive_rewards = env.cfg.rewards.only_positive_rewards
        #2. Iterating Over Reward Configurations
        for name, params in env.cfg.rewards.__dict__.items():
            #2.1 Remove zero scales + Multiply non-zero ones by dt
            if not isinstance(params, dict) or params["scale"] == 0:
                continue
            params["scale"] *= env.dt
            function = params["func"]
            #2.2 Handling Degrees of Freedom (DOFs) and Bodies as Reward Parameters
            #Retrieve the DOF Indices and Body Indices from the Robot
            if "dofs" in params.keys():
                params["dof_indices"], _ = env.robot.find_dofs(params["dofs"])
            if "bodies" in params.keys():
                params["body_indices"], _ = env.robot.find_bodies(params["bodies"])
            #2.3 Storing Reward Functions and Parameters
            self.reward_functions[name] = function
            self.reward_params[name] = params
            #2.4 Enabling Sensors
            if params.get("sensor") is not None:
                env.enable_sensor(params["sensor"])

        #3. Initializing Episode Sums
        #For each reward, it creates a PyTorch tensor initialized to zeros with the shape (env.num_envs,)
        self.episode_sums = {
            name: torch.zeros(
                env.num_envs,
                dtype=torch.float,
                device=env.device,
                requires_grad=False,
            )
            for name in self.reward_functions.keys()
        }

    def compute_reward(self, env):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self.__init__())
        adds each terms to the episode sums and to the total reward
        """
        #1. Resetting Total Reward
        self.reward = 0.0
        #2. Iterating Over Reward Functions
        for name, function in self.reward_functions.items():
            if name == "termination":
                continue  # handled separately after clipping
            params = self.reward_params[name]
            rew = function(env, params) * params["scale"]
            self.reward += rew
            self.episode_sums[name] += rew
        #3. Clipping Rewards
        if self.only_positive_rewards:
            self.reward = self.reward.clip(min=0.0)
        #4. Add termination reward 
        if "termination" in self.reward_functions:
            params = self.reward_params["termination"]
            rew = self.reward_functions["termination"](env, params) * params["scale"]
            self.reward += rew
            self.episode_sums["termination"] += rew
        return self.reward

    def log_info(self, env, env_ids, extras_dict):
        """ Fill env extras with episode sum of each reward """
        for key in self.episode_sums.keys():
            extras_dict["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / env.max_episode_length_s  # FIXME
            self.episode_sums[key][env_ids] = 0.0

    def remove_reward(self, name:str):
        # self.reward_functions.pop(name)
        self.reward_params[name]["scale"] = 0.
        # self.episode_sums.pop(name)