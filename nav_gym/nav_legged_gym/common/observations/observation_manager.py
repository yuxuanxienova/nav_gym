#isaac

#python
import torch
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
class ObsManager:
    def __init__(self, env):
        #1. Initializing Attributes
        self.obs_per_group = {}
        self.obs_dims_per_group = {}
        #-----------------------
        self.func_name_to_func = {}
        self.func_to_func_name = {}
        #-------------------------
        self.obs = {}
        self.obs_per_func = {}
        if class_to_dict(env.cfg).get("observations", None) is None:
            return
        obs_cfg_dict =class_to_dict(env.cfg.observations)
        #2. Iterating Over Observation Groups
        for group_name, obs_group_dict in obs_cfg_dict.items():
            if obs_group_dict is None:
                continue
            #2.1 Initializing Observation Group
            self.obs_per_group[group_name] = []
            obs_dim = 0
            add_noise = obs_group_dict["add_noise"]
            #2.2 Iterating Over Observation Functions in the Group
            for func_name, params in obs_group_dict.items():
                if not isinstance(params, dict):
                    continue
                if not add_noise:  # turn off all noise
                    params["noise"] = None
                #Retrieving the Observation Function
                function = params["func"]
                #Handling Degrees of Freedom (DOFs) and Bodies as Parameters
                #Retrieve the DOF Indices and Body Indices from the Robot
                if "dofs" in params.keys():
                    params["dof_indices"], _ = env.robot.find_dofs(params["dofs"])
                if "bodies" in params.keys():
                    params["body_indices"], _ = env.robot.find_bodies(params["bodies"])
                #Storing Observation Function and Parameters
                self.obs_per_group[group_name].append((function, params))
                #Enabling Sensors if Required
                # if params.get("sensor") is not None:
                #     env.enable_sensor(params["sensor"])
                #Accumulating Observation Dimensions
                obs_dim += function(env, params).shape[1]
                #-----------------------
                self.func_name_to_func[func_name] = function
                self.func_to_func_name[function] = func_name
                #-----------------------------
            self.obs_dims_per_group[group_name] = obs_dim
        # self.compute_obs(env)

    def compute_obs(self, env, obs_group=None):
        #1. Resetting Observations
        self.obs = {}
        #2. Determining Which Groups to Compute
        if obs_group is None:
            iterator = self.obs_per_group.items()
        else:
            iterator = [(obs_group, self.obs_per_group[obs_group])]
        #3. Iterating Over Selected Observation Groups
        for group, function_list in iterator:
            obs_list = []
            #3.1 Iterating Over Observation Functions in the Group
            for function, params in function_list:
                obs = function(env, params)
                noise = params.get("noise")
                clip = params.get("clip")
                scale = params.get("scale")
                #Applying Noise to Observations
                if noise:
                    obs = self._add_uniform_noise(obs, noise)
                #Clipping Observations
                if clip is not None:
                    obs = obs.clip(min=clip[0], max=clip[1])
                #Scaling Observations
                if scale is not None:
                    obs *= scale
                # NaN Check After Processing
                if torch.isnan(obs).any():
                    print(f"NaN detected in observation '{function.__name__}' within group '{group}'.")
                    # Optional: Print more detailed information
                    nan_indices = torch.isnan(obs).nonzero(as_tuple=False)
                    print(f"NaN indices in '{function.__name__}': {nan_indices}")
                    # Optional: Handle the NaN (e.g., replace, abort, etc.)
                    # For example, you can replace NaNs with zeros:
                    # obs = torch.nan_to_num(obs, nan=0.0)
                #Collecting Processed Observations
                obs_list.append(obs)
                #-----------------------
                self.obs_per_func[self.func_to_func_name[function]] = obs
                #-----------------------
            #3.2 Concatenating Observations Within the Group
            self.obs[group] = torch.cat(obs_list, dim=1)
        return self.obs

    def _add_uniform_noise(self, obs, noise_level):
        return obs + (2 * torch.rand_like(obs) - 1) * noise_level
    def get_obs_dims_from_group(self, group_name):
        if group_name in self.obs_dims_per_group:
            return self.obs_dims_per_group[group_name]
        else:
            return None
    def get_obs_from_group(self,group_name:str):
        if group_name in self.obs:
            return self.obs[group_name]
        else:
            return None

if __name__ == "__main__":
    cfg = LocomotionEnvCfg()
    print(cfg.observations)

