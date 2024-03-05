from typing import Any, Callable, Literal

import torch
from torch import nn

from .act import GaussianMF


class ANN(nn.Module):
    """
    A PyTorch implementation of an artificial neural network.

    An ANN is a machine learning model inspired by the structure and function of the human brain,
    consisting of interconnected nodes that process and transmit information.
    This implementation includes the option to use biases and layer normalization,
    and to apply dropout as a regularization technique.
    """

    def __init__(
            self,
            d_input: int,
            d_output: int,
            d_hidden: int = 128,
            n_hidden_layers: int = 0,
            hidden_activation: (
                    Callable[[Any], Any] |
                    Literal['tanh', 'celu', 'relu', 'leaky_relu', 'gelu', 'elu', 'hardtanh', 'gauss', 'none'] |
                    str) = None,
            bias: bool = True,
            layer_norm: bool = False,
            layer_norm_eps: float = 1e-5,
            dropout_rate: float = 0.1,
            **kwargs
    ):
        """
        Args:
            d_input: The input dimension of the data.
            d_output: The output dimension of the data.
            d_hidden: The number of units in the hidden layers. Default: 128.
            n_hidden_layers: The number of hidden layers in the ANN. Default: 1.
            hidden_activation: The activation function to use in the hidden layers.
                Accepts a callable function or one of the following strings:
                'tanh', 'celu', 'relu', 'leaky_relu', 'gelu', 'elu', 'hardtanh', or 'none'. Default: None.
            bias: Whether to include biases in the layers. Default: True.
            layer_norm: Whether to apply layer normalization to the layers. Default: False.
            layer_norm_eps: The epsilon value to use for layer normalization. Default: 1e-5.
            dropout_rate: The probability of dropping out a unit in the dropout layer. Default: 0.1.
        """
        super().__init__()

        for kwarg in kwargs:
            print(f'Unknown keyword argument: {kwarg}={kwargs[kwarg]}')

        if isinstance(hidden_activation, str):
            hidden_activation = hidden_activation.lower()
        if hidden_activation is None:
            hidden_activation = 'none'

        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.n_hidden_layers = n_hidden_layers
        self.bias = bias
        self.layer_norm = layer_norm
        self.layer_norm_eps = layer_norm_eps if self.layer_norm else 0
        self.hidden_activation = hidden_activation
        if isinstance(self.hidden_activation, str):
            if self.hidden_activation == 'none':
                self.hidden_activation = lambda x: x
            else:
                if self.hidden_activation == 'gauss':
                    self.hidden_activation = GaussianMF(elementwise_affine=True)
                    self.hidden_activation.__name__ = 'gauss'
                else:
                    try:
                        self.hidden_activation = getattr(torch, self.hidden_activation)
                    except AttributeError:
                        self.hidden_activation = getattr(nn.functional, self.hidden_activation)

        layers = []
        norms = []
        if n_hidden_layers <= 0:
            layers.append(nn.Linear(d_input, d_output, bias=bias))
        else:
            layers.append(nn.Linear(d_input, d_hidden, bias=bias))
            layers += [nn.Linear(d_hidden, d_hidden, bias=bias) for _ in range(n_hidden_layers - 1)]
            if layer_norm:
                norms += [nn.LayerNorm(d_hidden, eps=layer_norm_eps) for _ in range(n_hidden_layers - 1)]
            layers.append(nn.Linear(d_hidden, d_output, bias=bias))

        self.layers = nn.ModuleList(layers)
        if layer_norm:
            self.norms = nn.ModuleList(norms)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        """
        Processes the input data through the neural network, making predictions based on the input data.

        Args:
            inputs (torch.Tensor): The input data to be processed through the neural network.

        Returns: Logits tensor.
        """

        y = inputs
        for idx, layer in enumerate(self.layers):
            y = layer(y)
            if idx < len(self.layers) - 1:
                if self.layer_norm:
                    y = self.norms[idx](y)
                y = self.hidden_activation(y)
                y = self.dropout(y)

        return y

    @property
    def config(self):
        return {
            'd_input': self.d_input,
            'd_output': self.d_output,
            'd_hidden': self.d_hidden,
            'n_hidden_layers': self.n_hidden_layers,
            'hidden_activation': self.hidden_activation.__name__,
            'bias': self.bias,
            'layer_norm': self.layer_norm,
            'layer_norm_eps': self.layer_norm_eps,
            'dropout': self.dropout.p
        }


class LayerNet(nn.Module):
    """
    A PyTorch implementation of an artificial neural network.

    An ANN is a machine learning model inspired by the structure and function of the human brain,
    consisting of interconnected nodes that process and transmit information.
    This implementation includes the option to use biases and layer normalization,
    and to apply dropout as a regularization technique.
    """

    def __init__(
            self,
            d_input: int,
            d_output: int,
            layers: list[int] = None,
            activations: list[
                Callable[[Any], Any] |
                Literal['tanh', 'celu', 'relu', 'leaky_relu', 'gelu', 'elu', 'hardtanh', 'none'] |
                str] = None,
            bias: bool | list[bool] = True,
            layer_norm: bool = False,
            layer_norm_eps: float = 1e-5,
            dropout_rate: float = 0.1,
            **kwargs
    ):
        super().__init__()

        for kwarg in kwargs:
            print(f'Unknown keyword argument: {kwarg}={kwargs[kwarg]}')

        activations = [
            act.lower() if isinstance(act, str)
            else 'none' if act is None
            else act
            for act in activations
        ]

        self.d_input = d_input
        self.d_output = d_output
        self.layers = layers
        if self.layers is None:
            self.layers = []

        self.layers = [d_input] + self.layers + [d_output]

        self.bias = bias
        if isinstance(self.bias, bool):
            self.bias = [self.bias] * (len(self.layers) - 1)
        assert len(self.bias) == len(self.layers) - 1, \
            'There should be exactly one bias configuration for each layer.'

        assert len(activations) == len(self.layers) - 2, \
            'There should be exactly one activation function configuration for each layer.'
        self.activations = []
        for act in activations:
            if isinstance(act, str):
                if act == 'none':
                    self.activations.append(lambda x: x)
                else:
                    try:
                        self.activations.append(getattr(torch, act))
                    except AttributeError:
                        self.activations.append(getattr(nn.functional, act))
            else:
                self.activations.append(act)

        self.nn = nn.ModuleList([
            nn.Linear(d_in, d_out, bias=bias)
            for d_in, d_out, bias in zip(self.layers[:-1], self.layers[1:], self.bias)
        ])

        self.layer_norm = layer_norm
        self.layer_norm_eps = layer_norm_eps if self.layer_norm else 0

        if self.layer_norm:
            self.normalization_layers = nn.ModuleList([
                nn.LayerNorm(dim, eps=layer_norm_eps)
                for dim in self.layers[1:-1]
            ])

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        """
        Processes the input data through the neural network, making predictions based on the input data.

        Args:
            inputs (torch.Tensor): The input data to be processed through the neural network.

        Returns: ANNOutput or dict: The output of the neural network, depending on the value of `return_dict`.
            If `return_dict` is False, returns an object of the `ANNOutput` class with the following attribute:
                - logits (torch.Tensor): The predictions made by the neural network.
            If `return_dict` is True, returns a dictionary with the following key-value pairs:
                - 'logits': (torch.Tensor) The predictions made by the neural network.
        """
        y = inputs
        for idx, layer in enumerate(self.nn):
            y = layer(y)
            if idx != len(self.nn) - 1:
                if self.layer_norm:
                    y = self.normalization_layers[idx](y)
                y = self.activations[idx](y)
                y = self.dropout(y)

        return y

    @property
    def config(self):
        return {
            'layers': self.layers,
            'activations': [
                act.__name__ if hasattr(act, '__name__') else 'none'
                for act in self.activations
            ],
            'bias': self.bias,
            'layer_norm': self.layer_norm,
            'layer_norm_eps': self.layer_norm_eps,
            'dropout': self.dropout.p
        }
