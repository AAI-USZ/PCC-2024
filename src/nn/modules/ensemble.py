import random
from typing import Literal, Iterable

import torch
from torch import nn

from .ann import ANN
from .mixin import TorchRegressor


class EnsembleEstimator(nn.Module):
    def __init__(
            self,
            d_output: int,
            estimators: Iterable,
            collector: Literal['conv', 'linear'] = 'conv',
            bias: bool = True,
            no_grad: bool = True
    ):
        super().__init__()

        if bias is None:
            bias = True
        if no_grad is None:
            no_grad = True

        estimators = [
            estim if isinstance(estim, nn.Module) else TorchRegressor(regressor=estim)
            for estim in estimators
        ]
        n_estimators = len(estimators)
        self.estimators = nn.ModuleList(estimators)
        self.no_grad = no_grad
        if collector == 'conv':
            self.collector = nn.Conv1d(n_estimators, 1, 1)
        elif collector == 'linear':
            self.collector = nn.Linear(n_estimators * d_output, d_output, bias=bias)
        else:
            raise TypeError(f'Collector of type {collector} is unknown.')

        self._collector_type = collector
        with torch.no_grad():
            if collector == 'conv':
                self.collector.weight.data = torch.ones(
                    self.collector.weight.shape, dtype=self.collector.weight.dtype)
                if bias:
                    self.collector.bias.data = torch.zeros(
                        self.collector.bias.shape, dtype=self.collector.bias.dtype)
            else:
                self.collector.weight.data = torch.eye(
                    d_output, dtype=self.collector.weight.dtype).repeat(1, n_estimators)
                if bias:
                    self.collector.bias.data = torch.zeros(
                        self.collector.bias.shape, dtype=self.collector.bias.dtype)

    def _forward_estimators(self, x):
        ys = []
        for estimator in self.estimators:
            logits = estimator(x)
            # in case of multi output, select first
            if isinstance(logits, tuple):
                logits = logits[0]
            ys.append(logits)
        return ys

    def forward(self, x):
        if self.no_grad:
            with torch.no_grad():
                ys = self._forward_estimators(x)
        else:
            ys = self._forward_estimators(x)

        ds = len(ys)
        if self._collector_type == 'conv':
            y = torch.stack(ys, dim=1)
            y = self.collector(y)
            y = y.squeeze(dim=1)
            y = y / ds
        elif self._collector_type == 'linear':
            y = torch.concat(ys, dim=-1)
            y = self.collector(y)
            y = y / ds
        else:
            y = torch.stack(ys, dim=1)
            y = self.collector(y)

        return y

    def parameters(self, recurse: bool = True):
        return self.collector.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
        return self.collector.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)

    @property
    def n_estimators(self):
        return len(self.estimators)


class EnsembleANN(nn.Module):
    def __init__(
            self,
            d_input: int,
            d_output: int,
            *,
            n_estimators: int = 5,
            bias: bool = True,
            dropout_rate: float = 0.1,
            seed: int = None
    ):
        super().__init__()

        if bias is None:
            bias = True
        if seed is not None:
            random.seed(seed)

        ann_models = []
        for i in range(n_estimators):
            d_hidden = random.randint(64, 384)
            n_hidden_layers = random.randint(0, 6)
            hidden_activation = random.choice(['gelu', 'celu', 'tanh', 'relu', 'none'])
            bias = random.random() < 0.5
            print(f'-- ---------------- {i} Param Config ---------------- --')
            print(f'd_hidden:           {d_hidden}')
            print(f'n_hidden_layers:    {n_hidden_layers}')
            print(f'hidden_activation:  {hidden_activation}')
            print(f'bias:               {bias}')
            print(f'-- ---------------- |||||||||||||||| ---------------- --')
            model = ANN(
                d_input, d_output, d_hidden=d_hidden, n_hidden_layers=n_hidden_layers,
                hidden_activation=hidden_activation, bias=bias, dropout_rate=dropout_rate)
            ann_models.append(model)

        self.estimators = nn.ModuleList(ann_models)
        self.aggr = nn.Conv1d(n_estimators, 1, 1)
        with torch.no_grad():
            self.aggr.weight.data = torch.ones((1, n_estimators, 1), dtype=self.aggr.weight.dtype)
            if bias:
                self.aggr.bias.data = torch.zeros((1,), dtype=self.aggr.bias.dtype)

    def forward(self, x):
        ys = []
        for estimator in self.estimators:
            logits = estimator(x)
            ys.append(logits)

        y = torch.stack(ys, dim=1)
        ds = y.shape[1]
        y = self.aggr(y)
        y = y.squeeze(dim=1)
        y = y / ds

        return y
