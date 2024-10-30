#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# isaacgym
from isaacgym import gymutil
# legged_gym
# from  import LEGGED_GYM_ROOT_DIR

# python
import collections.abc
import os
import copy
import torch
import numpy as np
import random
import argparse


"""
Dictionary operations.
"""


def update_dict(orig_dict: dict, new_dict: collections.abc.Mapping) -> dict:
    """Updates existing dictionary with values from a new dictionary.

    This function mimics the dict.update() function. However, it works for
    nested dictionaries as well.

    Reference:
        https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    Args:
        orig_dict (dict): The original dictionary to insert items to.
        new_dict (collections.abc.Mapping): The new dictionary to insert items from.

    Returns:
        dict: The updated dictionary.
    """
    for key, value in new_dict.items():
        if isinstance(value, collections.abc.Mapping):
            orig_dict[key] = update_dict(orig_dict.get(key, {}), value)
        else:
            orig_dict[key] = value
    return orig_dict


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionary."""
    if type(val) == dict:
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        print(val)


"""
Parsing configurations.
"""


# def get_args() -> argparse.Namespace:
#     """Defines custom command-line arguments and parses them.

#     Returns:
#         argparse.Namespace: Parsed CLI arguments.
#     """
#     custom_parameters = [
#         {
#             "name": "--task",
#             "type": str,
#             "default": "anymal_c_flat",
#             "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
#         },
#         {
#             "name": "--resume",
#             "action": "store_true",
#             "default": False,
#             "help": "Resume training from a checkpoint",
#         },
#         {
#             "name": "--experiment_name",
#             "type": str,
#             "help": "Name of the experiment to run or load. Overrides config file if provided.",
#         },
#         {
#             "name": "--run_name",
#             "type": str,
#             "help": "Name of the run. Overrides config file if provided.",
#         },
#         {
#             "name": "--load_run",
#             "type": str,
#             "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
#         },
#         {
#             "name": "--checkpoint",
#             "type": int,
#             "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
#         },
#         {
#             "name": "--headless",
#             "action": "store_true",
#             "default": False,
#             "help": "Force display off at all times",
#         },
#         {
#             "name": "--rl_device",
#             "type": str,
#             "default": "cuda:0",
#             "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
#         },
#         {
#             "name": "--num_envs",
#             "type": int,
#             "help": "Number of environments to create. Overrides config file if provided.",
#         },
#         {
#             "name": "--seed",
#             "type": int,
#             "help": "Random seed. Overrides config file if provided.",
#         },
#         {
#             "name": "--max_iterations",
#             "type": int,
#             "help": "Maximum number of training iterations. Overrides config file if provided.",
#         },
#         {
#             "name": "--log_dir",
#             "type": str,
#             "default": os.path.join(LEGGED_GYM_ROOT_DIR, "logs"),
#             "help": "Log directory. Defaulf: <LEGGED_GYM_ROOT_DIR>/logs. Set to 'None' to disable logging",
#         },
#     ]
#     # parse arguments
#     args = gymutil.parse_arguments(description="RL Policy using IsaacGym", custom_parameters=custom_parameters)
#     # name alignment
#     args.sim_device_id = args.compute_device_id
#     args.sim_device = args.sim_device_type
#     if args.sim_device == "cuda":
#         args.sim_device += f":{args.sim_device_id}"
#     return args


"""
MDP-related operations.
"""


def set_seed(seed: int):
    """Set the seeding of the experiment.

    Note:
        If input is -1, then a random integer between (0, 10000) is sampled.

    Args:
        seed (int): The seed value to set.
    """
    # default argument
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("[Utils] Setting seed: {}".format(seed))
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


"""
Model loading and saving.
"""


def get_load_path(root: str, load_run=-1, checkpoint: int = -1) -> str:
    # check if runs present in directory
    try:
        runs = os.listdir(root)
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except IndexError:
        raise ValueError(f"No runs present in directory: {root}")
    # path to the directory containing the run
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)
    # name of model checkpoint
    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    return os.path.join(load_run, model)


def export_policy_as_jit(actor_critic, normalizer, path, filename="policy.pt"):
    policy_exporter = TorchPolicyExporter(actor_critic, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(actor_critic, normalizer, path, filename="policy.onnx"):
    policy_exporter = OnnxPolicyExporter(actor_critic, normalizer)
    policy_exporter.export(path, filename)


class TorchPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer(
                "hidden_state",
                torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
            )
            self.register_buffer(
                "cell_state",
                torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
            )
            self.forward = self.forward_lstm
            self.reset = self.reset_memory

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class OnnxPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        return self.actor(x.squeeze(0)), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=False,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=True,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
