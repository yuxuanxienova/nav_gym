import torch
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
# solves circular imports of LeggedRobot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nav_gym.nav_legged_gym.envs.legged_env import LeggedEnv


class TerminationManager:
    def __init__(self, env: "LeggedEnv"):
        """Prepares a list of functions"""
        #1. Initializing Attributes
        self.functions = {}
        self.params = {}
        self.time_out_buf = torch.zeros_like(env.reset_buf)
        self.reset_on_termination = True
        if class_to_dict(env.cfg).get("terminations", None) is None:
            return
        #2. Retrieves the reset_on_termination flag from the environment's terminations configuration.
        #This flag determines whether the environment should reset when a termination condition is met.
        self.reset_on_termination = env.cfg.terminations.reset_on_termination
        #3. Iterating Over Termination Configurations
        for name, params in class_to_dict(env.cfg.terminations).items():
            if not isinstance(params, dict):
                continue
            #3.1 Handling Degrees of Freedom (DOFs) and Bodies as Parameters
            #Retrieve the DOF Indices and Body Indices from the Robot
            if "dofs" in params.keys():
                params["dof_indices"], _ = env.robot.find_dofs(params["dofs"])
            if "bodies" in params.keys():
                params["body_indices"], _ = env.robot.find_bodies(params["bodies"])

            #3.2 Function Retrieval
            function = params["func"]
            self.functions[name] = function
            self.params[name] = params
            #3.3 Enabling Sensors if Required
            if params.get("sensor") is not None:
                env.enable_sensor(params["sensor"])
        #4. initializing time counter
        self.time_counter = torch.zeros(env.num_envs, device=env.device)
    def check_termination(self, env: "LeggedEnv"):
        """Check terminations
        Calls each termiantion function which was defined in the config (processed in self.__init__). Returns the logical OR of all functions.
        """
        #1. Initializing Termination Buffers
        terminated = torch.zeros_like(env.reset_buf)
        self.time_out_buf[:] = 0
        #2. Iterating Over Termination Functions
        for name, function in self.functions.items():
            params = self.params[name]
            out = function(env, params)
            terminated |= out
            if params.get("time_out", False): # record time-outs separately
                self.time_out_buf |= out
        #3. Updating Time Counters
        self.time_counter += env.dt
        return terminated
    def log_info(self, env, env_ids, extras_dict):
        time_counter_termination = self.time_counter[env_ids][self.time_counter[env_ids] < env.max_episode_length_s]
        if time_counter_termination.numel() == 0:
            extras_dict["avg_termination_time"] = env.max_episode_length_s
        else:
            extras_dict["avg_termination_time"] = torch.mean(time_counter_termination)
        self.time_counter[env_ids] = 0.0