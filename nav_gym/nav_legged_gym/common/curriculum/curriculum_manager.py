import torch
# solves circular imports of LeggedRobot
from typing import TYPE_CHECKING
from nav_gym.nav_legged_gym.utils.conversion_utils import class_to_dict
if TYPE_CHECKING:
    from nav_gym.nav_legged_gym.envs.locomotion_env import LocomotionEnv


class CurriculumManager:
    def __init__(self, env: "LocomotionEnv"):
        """Prepares a list of fucntions"""
        #1. Initializing Attributes
        self.functions = {"on_reset": {}, "on_command_resample": {}}
        self.params = {}
        if class_to_dict(env.cfg).get("curriculum", None) is None:
            return
        #2. Registering Curriculum Functions
        #Iterates over each curriculum configuration defined in env.cfg.curriculum and registers the corresponding functions and parameters.
        for name, params in class_to_dict(env.cfg.curriculum).items():
            if not isinstance(params, dict):
                continue
            #2.1 Retrieves the mode from params. If not specified, defaults to "on_reset"
            mode = params.get("mode", "on_reset")
            function = params["func"]
            #2.2 Registers the function and parameters under the specified mode
            self.functions[mode][name] = function
            self.params[name] = params
            #2.3 Ensures that any necessary sensors are active
            # if params.get("sensor") is not None:
            #     env.enable_sensor(params["sensor"])

    def update_curriculum(self, env: "LocomotionEnv", env_ids, mode="on_reset"):
        """Update curriculum
        Calls each update function which was defined in the config (processed in self.__init__). Each function modifies the env directly.
        """
        for name, function in self.functions[mode].items():
            params = self.params[name]
            function(env, env_ids, params)

    def log_info(self, env: "LocomotionEnv", env_ids, extras_dict):
        # if "terrain_levels" in self.params.keys():
        #     extras_dict["terrain_level"] = torch.mean(env.terrain.terrain_levels.float())
        if "max_lin_vel_command" in self.params.keys():
            extras_dict["max_command_x"] = env.command_ranges["lin_vel_x"][1]
            extras_dict["max_command_y"] = env.command_ranges["lin_vel_y"][1]
