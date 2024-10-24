from typing import Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int],
        activation: str = "elu",
        last_activation: str = None,
    ):
        """Multi-layer perceptron.

        Args:
            input_dim (int): Dimension of the input.
            output_dim (int): Dimension of the output.
            hidden_dims (Tuple[int]): Dimensions of the hidden layers.
            activation (str, optional): Activation function. Defaults to "elu".
            last_activation (str, optional): Activation function of the last layer. Defaults to None.
        """
        super(MLP, self).__init__()
        activation = get_activation(activation)

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)

        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_index], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                layers.append(activation)

        if last_activation is not None:
            layers.append(get_activation(last_activation))

        self.sequential = nn.Sequential(*layers)

    def init_weights(self, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in self.sequential if isinstance(mod, nn.Linear))
        ]

    def forward(self, x):
        return self.sequential(x)

    def __str__(self):
        return f"{self.sequential}"


def get_activation(act_name: str):
    """Return the activation function given its name.

    Args:
        act_name (str): Name of the activation function.

    Returns:
        nn.Module: The activation function.

    Raises:
        ValueError: If the activation function is not found.
    """

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
        raise ValueError(f"Activation function {act_name} not found!")
