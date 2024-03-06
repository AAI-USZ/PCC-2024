from typing import Callable, Any, Literal

import torch
from torch import nn


class RecurrentNet(nn.Module):
    def __init__(
            self,
            d_input: int,
            d_output: int,
            *,
            d_hidden: int = 128,
            n_hidden_layers: int = 1,
            rnn_type: str | nn.RNNBase = 'gru',
            hidden_activation: (
                    Callable[[Any], Any] |
                    Literal['tanh', 'celu', 'relu', 'leaky_relu', 'gelu', 'elu', 'hardtanh', 'silu', 'none'] |
                    str) = None,
            bidirectional: bool = False,
            max_length: int = 1,
            aggregation: Literal['conv', 'min', 'max', 'sum', 'mean', 'none'] = None,
            bias: bool = True,
            dropout_rate: float = 0.1,
            **kwargs
    ):
        super().__init__()

        for kwarg in kwargs:
            print(f'Unknown keyword argument: {kwarg}={kwargs[kwarg]}')

        if isinstance(hidden_activation, str):
            hidden_activation = hidden_activation.lower()
        if hidden_activation is None:
            hidden_activation = 'none'
        if aggregation is not None:
            aggregation = aggregation.lower()
            if aggregation not in {'min', 'max', 'sum', 'mean', 'conv', 'none'}:
                raise TypeError(
                    f'Parameter aggregation should be one of '
                    f'"min", "max", "sum", "mean", "dense", "conv", "none", or None, and was {aggregation}')
        if d_hidden is None:
            d_hidden = 128
        if n_hidden_layers is None:
            n_hidden_layers = 1

        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden
        self.n_hidden_layers = n_hidden_layers
        self.bidirectional = bidirectional
        self.max_length = max_length
        self.aggregation = aggregation
        if self.aggregation == 'none':
            self.aggregation = None
        self.bias = bias
        self.hidden_activation = hidden_activation
        if isinstance(self.hidden_activation, str):
            if self.hidden_activation == 'none':
                self.hidden_activation = lambda x: x
            else:
                try:
                    self.hidden_activation = getattr(torch, self.hidden_activation)
                except AttributeError:
                    self.hidden_activation = getattr(nn.functional, self.hidden_activation)
        self.rnn_type = rnn_type
        if isinstance(self.rnn_type, str):
            self.rnn_type = self.rnn_type.lower()
            if self.rnn_type == 'rnn':
                rnn_cls = nn.RNN
            elif self.rnn_type == 'gru':
                rnn_cls = nn.GRU
            elif self.rnn_type == 'lstm':
                rnn_cls = nn.LSTM
            else:
                raise TypeError('rnn_type most be one of "rnn", "gru", "lstm"')
        elif not isinstance(self.rnn_type, str):
            rnn_cls = self.rnn_type
            self.rnn_type = self.rnn_type.__class__.__name__.lower()
        else:
            raise TypeError('rnn_type most be an instance of str, or RNNBase')

        mult = 2 if bidirectional else 1
        self.rnn = rnn_cls(
            d_input, d_hidden, num_layers=n_hidden_layers, bias=bias, batch_first=True, bidirectional=bidirectional)
        if isinstance(aggregation, str):
            aggregation = aggregation.lower()
            if aggregation == 'none':
                aggregation = None

        self.series_aggr = aggregation
        if self.series_aggr is not None:
            if self.series_aggr == 'conv':
                self.series_aggr = nn.Conv1d(max_length, 1, 1)
            else:
                fn = getattr(torch, self.series_aggr)
                self.series_aggr = lambda x: fn(x, dim=1, keepdim=True)
        self.output_layer = nn.Linear(d_hidden * mult, d_output)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, return_rnn_hidden=False):
        y = inputs

        rnn_out = self.rnn(y)
        y = rnn_out[0]
        y = self.hidden_activation(y)
        y = self.dropout(y)

        if self.series_aggr is not None:
            y = self.series_aggr(y)

        y = self.output_layer(y)

        if return_rnn_hidden:
            return y, *rnn_out[1:]
        return y
