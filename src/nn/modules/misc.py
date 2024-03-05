import itertools
from typing import Iterable

import torch
from torch import nn

from .act import GaussianMF


class FormulaNet(nn.Module):
    def __init__(self, n_dependent: int):
        super().__init__()

        self.n_dependent = n_dependent
        self.cross_w = nn.Parameter(torch.zeros(n_dependent ** 2))
        self.cross_b = nn.Parameter(torch.zeros(n_dependent ** 2))

        self.lin_w = nn.Parameter(torch.zeros(n_dependent))
        self.lin_b = nn.Parameter(torch.zeros(n_dependent))

        self.f = nn.ModuleList([GaussianMF(elementwise_affine=True) for _ in range(n_dependent ** 2)])

    def forward(self, x):
        y = x[..., None] * x[..., None, :]
        y = y.view(-1, self.n_dependent ** 2)

        for idx, gf in enumerate(self.f):
            y[:, idx] = gf(y[:, idx])

        y = self.cross_w * y + self.cross_b

        ly = self.lin_w * x + self.lin_b

        yo = y.sum(-1, keepdims=True) + ly.sum(-1, keepdims=True)

        return {
            'logits': yo
        }

    def get_formula(self, x: Iterable[str] = None):
        x_vars = [f'x{n}' for n in range(1, self.n_dependent + 1)]
        if x is not None:
            x_vars = [xn for xn in x]
            assert len(x_vars) == self.n_dependent

        # Create dependent parts
        # -- -------------------
        x_cross = [f'{cx1}*{cx2}' for cx1, cx2 in itertools.product(x_vars, x_vars)]

        # gaussian functions: f(x) = a * exp(-(x - b)^2 / 2c^2),
        x_fn_vars = [
            f'{fn.a.item()}*exp(-({x} - ({fn.b.item()}))^2 / 2*({fn.c.item()})^2)'
            for x, fn in zip(x_cross, self.f)
        ]

        x_dependent_vars = [
            f'{w.item()}*({x}) + ({b.item()})'
            for x, w, b in zip(x_fn_vars, self.cross_w, self.cross_b)
        ]

        # Create independent parts
        # -- ---------------------
        x_independent_vars = [
            f'{w.item()}*({x}) + ({b.item()})'
            for x, w, b in zip(x_vars, self.lin_w, self.lin_b)
        ]

        # Aggregate parts
        # -- ------------

        dep_sum = ' + '.join(x_dependent_vars)
        indep_sum = ' + '.join(x_independent_vars)

        fun = f'({dep_sum}) + ({indep_sum})'

        return fun
