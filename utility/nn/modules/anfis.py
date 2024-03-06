from typing import TypedDict, NotRequired, Any, Literal

import numpy as np
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch import nn

from .act import BellMF, GaussianMF, SechMF, WoaMF, BumpMF, SigmoidMF, TriangleMF, TrapezoidalMF


def _get_cls_from_f(f: str):
    if f == 'bell':
        cls = BellMF
    elif f == 'gaussian':
        cls = GaussianMF
    elif f == 'sech':
        cls = SechMF
    elif f == 'woa':
        cls = WoaMF
    elif f == 'bump':
        cls = BumpMF
    elif f == 'sigmoid':
        cls = SigmoidMF
    elif f == 'tri':
        cls = TriangleMF
    elif f == 'trap':
        cls = TrapezoidalMF
    else:
        raise TypeError(f'Unknown function requested: {f}')
    return cls


class FuzzyFunctionConfig(TypedDict):
    function: Literal['bell', 'gaussian', 'sech', 'woa', 'bump', 'sigmoid', 'tri', 'trap']
    n_members: int
    elementwise_affine: NotRequired[bool]
    params: NotRequired[dict[str, Any]]


class FuzzyFunction(nn.Module):
    """
    Fuzzification function who can live inside a FuzzyLayer.
    """

    def __init__(
            self,
            f: str | nn.Module,
            n_members: int = 5,
            elementwise_affine: bool = None,
            *args,
            **kwargs
    ):
        """
        Represents a generalized fuzzy function.

        The class requires a function to be used during fuzzyfication.

        Args:
            f: fuzzyfication function
            n_members: number of member function
            elementwise_affine: function should have parameters as learnable parameters
            *args: additional arguments to be passed to the fuzzy function `f`
            **kwargs: additional keyword arguments to be passed to the fuzzy function `f`
        """

        super().__init__()
        if isinstance(f, str):
            cls = _get_cls_from_f(f)
            self.f = cls(elementwise_affine=elementwise_affine, *args, **kwargs)
        else:
            self.f = f
        self.n_members = n_members

    @classmethod
    def from_config(cls, config_or_instance):
        if isinstance(config_or_instance, dict):
            params = config_or_instance.get('params', {})
            return cls(
                f=config_or_instance['function'], n_members=config_or_instance['n_members'],
                elementwise_affine=config_or_instance.get('elementwise_affine'), **params)
        if type(config_or_instance) is FuzzyFunction:
            return config_or_instance
        raise TypeError(
            'Config or instance should be either of type `FuzzyFunc` (dict) or `FuzzyFunction`. '
            f'{type(config_or_instance)}')

    def forward(self, x, *args, **kwargs):
        return torch.repeat_interleave(
            self.f(x, *args, **kwargs),
            self.n_members, dim=-1
        ).view(*x.shape, self.n_members)


class NFuzzyFunction(nn.Module):
    """
    Fuzzyfication function with N separate membership functions,
    who can live inside a FuzzyLayer.
    """

    def __init__(
            self,
            f: str,
            n_members: int = 5,
            elementwise_affine: bool = None,
            *args,
            **kwargs
    ):
        """
        Represents a generalized fuzzy function.

        The class requires a function to be used during fuzzyfication.

        Args:
            f: fuzzyfication function
            n_members: number of member function
            elementwise_affine: function should have parameters as learnable parameters
            *args: additional arguments to be passed to the fuzzy function `f`
            **kwargs: additional keyword arguments to be passed to the fuzzy function `f`
        """

        super().__init__()
        cls = _get_cls_from_f(f)
        self.f = nn.ModuleList([
            cls(elementwise_affine=elementwise_affine, *args, **kwargs)
            for _ in range(n_members)
        ])
        self.n_members = n_members

    @classmethod
    def from_config(cls, config_or_instance):
        if isinstance(config_or_instance, dict):
            params = config_or_instance.get('params', {})
            return cls(
                f=config_or_instance['function'], n_members=config_or_instance['n_members'],
                elementwise_affine=config_or_instance.get('elementwise_affine'), **params)
        if type(config_or_instance) is NFuzzyFunction:
            return config_or_instance
        raise TypeError(
            'Config or instance should be either of type `FuzzyFunc` (dict) or `FuzzyFunction`. '
            f'{type(config_or_instance)}')

    def forward(self, x, *args, **kwargs):
        y = torch.repeat_interleave(x, self.n_members, dim=-1).view(*x.shape, self.n_members)
        for idx, f in enumerate(self.f):
            y[..., idx] = f(y[..., idx], *args, **kwargs)
        return y


class FuzzyLayer(nn.Module):
    """
    Represents a fuzzy layer. Can be used inside ANFIS model.

    This layer's purpose is to fuzzify the inputs, based on the given membership functions.

    Attributes:
        fuzzy: module list of fuzzy layers
    """

    def __init__(
            self,
            fuzzy_functions: list[FuzzyFunctionConfig | FuzzyFunction],
            mode: str = None
    ):
        super().__init__()

        if mode is None:
            mode = 'same'
        else:
            mode = mode.lower()

        if mode == 'unique':
            self.fuzzy = nn.ModuleList([
                NFuzzyFunction.from_config(ff)
                for ff in fuzzy_functions
            ])
        elif mode == 'same':
            self.fuzzy = nn.ModuleList([
                FuzzyFunction.from_config(ff)
                for ff in fuzzy_functions
            ])
        else:
            raise TypeError(f'Unknown mode {mode} passed. Valid modes: same, unique.')

    def forward(self, x):
        # TODO: this is potentially bad code --> we do not necessarily iterate through every input.
        #  It is only nice behaving when there are as many dependent variables as fuzzy functions.
        return [fuzz(x[..., i]) for i, fuzz in enumerate(self.fuzzy)]


class RuleLayer(nn.Module):
    """
    Rule layer. Can be used inside ANFIS.
    """

    def __init__(self, n_members: int = None):
        super().__init__()
        if n_members is None or n_members <= 0:
            n_members = -1
        self.n_members = n_members

    def _set_n_members_if_not_set(self, n):
        if n <= 0:
            raise ValueError('Number of rules must be greater than zero.')

        if self.n_members == -1:
            self.n_members = n

    def _gen_view_indices(self, idx):
        view = torch.ones(self.n_members, dtype=torch.int)
        view[idx] = -1
        return view

    def forward(self, x: list | tuple):
        if len(x) <= 0:
            return x

        assert all(len(a) == len(b) for a, b in zip(x[:-1], x[1:])), \
            'All inputs should have the same batch size.'

        dim = x[0].dim()
        self._set_n_members_if_not_set(len(x))

        # get main shape of data
        # batched data
        if dim == 2:
            main_shape = x[0].shape[:1]
        # batched sequential data
        elif dim == 3:
            main_shape = x[0].shape[:2]
        else:
            raise TypeError('Data dimension should be either 2 or 3.')

        # views to help creating all possible combinations
        fuzz_views = [
            fuzz_part.view(*main_shape, *self._gen_view_indices(idx))
            for idx, fuzz_part in enumerate(x)
        ]
        output = 1
        for fuzz in fuzz_views:
            output = output * fuzz

        output = output.reshape(*main_shape, -1)
        return output


class ConsequenceLayer(nn.Module):
    """
    Consequence layer. Can be used inside ANFIS.

    This layer is basically a wrapper around normalization,
    and a linear layer.

    During forward pass, the data will we normalized first,
    then passed through the linear layer.
    """

    def __init__(self, d_input: int, n_rules: int, bias: bool = None):
        if bias is None:
            bias = True

        super().__init__()
        self.bias = bias
        self.linear = nn.Linear(d_input, n_rules, bias=self.bias)

    def forward(self, x):
        y = self.linear(x)
        return y


class FISAggregator(nn.Module):
    def __init__(self, aggregation_function: str = None):
        if isinstance(aggregation_function, str) and aggregation_function not in {'sum', 'mean', 'prod', 'none'}:
            raise TypeError(
                f'Unknown aggregation function {aggregation_function}. '
                f'Accepted values are: {["sum", "mean", "prod", "none"]}')

        super().__init__()

        if aggregation_function is not None and aggregation_function != 'none':
            self._f = getattr(torch, aggregation_function)
        else:
            self._f = lambda x, *args, **kwargs: x

    def forward(self, x):
        return self._f(x, dim=-1, keepdim=True)


class ANFIS(nn.Module):
    def __init__(
            self,
            d_input: int,
            member_functions: list[FuzzyFunctionConfig | FuzzyFunction],
            bias: bool = None,
            fuzzy_mode: Literal['unique', 'same'] = None,
            aggregation_function: Literal['sum', 'prod', 'mean', 'none'] | None = 'sum',
            normalize: bool = None,
            **kwargs
    ):
        """
        ANFIS implementation.

        Args:
            d_input: Input dimension.
            member_functions: Description of membership functions.
            bias: Use bias in consequent layer. Default: True.
            fuzzy_mode: Fuzzifycation mode. ('same'|'unique')
            aggregation_function ('sum', 'prod', 'mean', 'none'): Aggregation function to use.
                De-fuzzyfication layer. ('sum'|'prod'|'mean'|'none'|None)
            normalize: Use normalization. Default value will be assumed from aggregation function.
                If aggregation function is not None, then by default normalization will be used (True),
                if there is no aggregation function, defaults to False.
        """

        super().__init__()

        assert len(kwargs) == 0, 'No keyword arguments accepted'

        self.d_input = d_input
        self.member_functions = member_functions
        self.bias = bias
        self.aggregation_function = aggregation_function
        self._normalize = normalize
        if normalize is None:
            if aggregation_function is not None and aggregation_function.lower() != 'none':
                self._normalize = True
            else:
                self._normalize = False

        self.fuzzy = FuzzyLayer(member_functions, mode=fuzzy_mode)
        self.rule = RuleLayer(n_members=len(member_functions))
        n_rules = np.prod([ff.n_members for ff in self.fuzzy.fuzzy])
        self.consequence = ConsequenceLayer(d_input=d_input, n_rules=n_rules, bias=bias)
        self.aggregator = FISAggregator(aggregation_function=aggregation_function)

    def forward(self, x, s=None):
        if s is None:
            s = x

        sy = self.fuzzy(s)
        sy = self.rule(sy)
        y = self.consequence(x)

        if self._normalize:
            sy = F.normalize(sy, p=1.0, dim=-1)

        y = sy * y
        y = self.aggregator(y)

        return y

    @property
    def n_members(self):
        return len(self.fuzzy.fuzzy)

    def n_rules(self):
        return np.prod([ff.n_members for ff in self.fuzzy.fuzzy])


class ANFISNet(nn.Module):
    """
    Hybrid model of an ANFIS and a neural network.

    ANFIS is used to fuzzify, the data and output fuzzy rules of inputs.
    The network will use the fuzzified inputs, eat them,
    and produce a resultant tensor based on its internal inner workings.

    Attributes:
        anfis (nn.Module): ANFIS part of the model
        net (nn.Module): Neural Network (NN) part of the model
    """

    def __init__(
            self,
            anfis: nn.Module,
            net: nn.Module = None,
    ):
        super().__init__()
        self.anfis = anfis
        self.net = net

    def forward(self, x, s=None, *args, **kwargs):
        if s is None:
            s = x

        y = self.anfis(x, s)
        if self.net is not None:
            y = self.net(y, *args, **kwargs)

        return y


class GaussNet(nn.Module):
    def __init__(self, d_input: int, d_output: int, nf: int):
        assert nf > 0

        super().__init__()
        self.gaussians = nn.ModuleList([
            GaussianMF(elementwise_affine=True)
            for _ in range(nf)
        ])
        self.linears = nn.ModuleList([
            nn.Linear(in_features=d_input, out_features=d_output)
            for _ in range(nf)
        ])

    def forward(self, x):
        y = [
            gf(x)
            for gf in self.gaussians
        ]
        y = [
            F.gelu(l(z))
            for l, z in zip(self.linears, y)
        ]
        val = 0.
        for v in y:
            val = v + val
        return {
            'logits': val
        }
