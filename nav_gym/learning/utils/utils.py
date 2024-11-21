# python
import os
import git
import pathlib
import numpy as np
from typing import Tuple
# torch
import torch


def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir, repositories):
    for repository_file_path in repositories:
        repo = git.Repo(repository_file_path, search_parent_directories=True)
        repo_name = pathlib.Path(repo.working_dir).name
        t = repo.head.commit.tree
        file_path = os.path.join(logdir, f"{repo_name}_git.diff")
        if not os.path.exists(file_path):
            content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
            with open(file_path, "x") as f:
                f.write(content)


def get_skew_matrix(vec):
    """Get the skew matrix of a vector."""
    matrix = torch.zeros(vec.shape[0], 3, 3, dtype=torch.float, requires_grad=False)
    matrix[:, 0, 1] = -vec[:, 2]
    matrix[:, 0, 2] = vec[:, 1]
    matrix[:, 1, 0] = vec[:, 2]
    matrix[:, 1, 2] = -vec[:, 0]
    matrix[:, 2, 0] = -vec[:, 1]
    matrix[:, 2, 1] = vec[:, 0]
    return matrix


def get_base_ang_vel_from_base_quat(base_quat, dt, target_frame="local"):
    """
    Get the base angular velocity from the base quaternion.
    args:
        base_quat:      torch.Tensor (num_trajs, num_steps, 4)
        dt:             float
    returns:
        base_ang_vel:   torch.Tensor (num_trajs, num_steps, 3) expressed in the target frame
    """
    num_trajs, num_steps, _ = base_quat.size()
    device = base_quat.device
    mapping = torch.zeros(num_trajs, num_steps, 3, 4, device=device, dtype=torch.float, requires_grad=False)
    mapping[:, :, :, -1] = -base_quat[:, :, :-1]
    if target_frame == "local":
        mapping[:, :, :, :-1] = get_skew_matrix(-base_quat[:, :, :-1].flatten(0, 1)).view(num_trajs, num_steps, 3, 3)
    elif target_frame == "global":
        mapping[:, :, :, :-1] = get_skew_matrix(base_quat[:, :, :-1].flatten(0, 1)).view(num_trajs, num_steps, 3, 3)
    else:
        raise ValueError(f"Unknown target frame {target_frame}")
    mapping[:, :, :, :-1] += torch.eye(3, device=device, dtype=torch.float, requires_grad=False).repeat(num_trajs, num_steps, 1, 1) * base_quat[:, :, -1].unsqueeze(-1).unsqueeze(-1)
    base_ang_vel = 2 * mapping[:, :-1, :, :] @ ((base_quat[:, 1:, :] - base_quat[:, :-1, :]) / dt).unsqueeze(-1)
    base_ang_vel = torch.cat((base_ang_vel[:, [0]], base_ang_vel), dim=1).squeeze(-1)
    return base_ang_vel


def get_base_quat_from_base_ang_vel(base_ang_vel, dt, source_frame="local", init_base_quat=None):
    """
    Get the base quaternion from the base angular velocity.
    args:
        base_ang_vel:   torch.Tensor (num_trajs, num_steps, 3) expressed in the source frame
        dt:             float
    returns:
        base_quat:      torch.Tensor (num_trajs, num_steps, 4)
    """
    num_trajs, num_steps, _ = base_ang_vel.size()
    device = base_ang_vel.device
    if init_base_quat is None:
        init_base_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float, requires_grad=False).repeat(num_trajs, 1)
    base_quat = torch.zeros(num_trajs, num_steps, 4, device=device, dtype=torch.float, requires_grad=False)
    base_quat[:, 0, :] = init_base_quat
    for step in range(num_steps - 1):
        base_quat_step = base_quat[:, step, :]
        mapping = torch.zeros(num_trajs, 3, 4, device=device, dtype=torch.float, requires_grad=False)
        mapping[:, :, -1] = -base_quat_step[:, :-1]
        if source_frame == "local":
            mapping[:, :, :-1] = get_skew_matrix(-base_quat_step[:, :-1])
        elif source_frame == "global":
            mapping[:, :, :-1] = get_skew_matrix(base_quat_step[:, :-1])
        else:
            raise ValueError(f"Unknown source frame {source_frame}")
        mapping[:, :, :-1] += torch.eye(3, device=device, dtype=torch.float, requires_grad=False).repeat(num_trajs, 1, 1) * base_quat_step[:, -1].unsqueeze(-1).unsqueeze(-1)
        base_ang_vel_step = base_ang_vel[:, step, :].unsqueeze(-1)
        quat_change_est = (0.5 * dt * mapping.transpose(-2, -1) @ base_ang_vel_step).squeeze(-1)
        base_quat[:, step + 1, :] = quat_change_est + base_quat_step
    return base_quat


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir, repositories):
    for repository_file_path in repositories:
        repo = git.Repo(repository_file_path, search_parent_directories=True)
        repo_name = pathlib.Path(repo.working_dir).name
        git_log_msg = repo.git.log(p=True, n=1)

        # content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)} "
        content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git log & diff ---\n{git_log_msg} "
        with open(os.path.join(logdir, f"{repo_name}_git.diff"), "x", encoding="utf-8") as f:
            f.write(content)

