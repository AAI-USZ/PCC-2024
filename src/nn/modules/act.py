import torch
from torch import nn

import simtools.nn.functional as rf


def _register_function_params(module: nn.Module, elementwise_affine: bool = None, **kwargs):
    if elementwise_affine is None:
        elementwise_affine = True

    for key, value in kwargs.items():
        if value is not None:
            if type(value) is not torch.Tensor:
                value = torch.tensor(value, dtype=torch.float)
            if elementwise_affine:
                value = nn.Parameter(value)
                module.register_parameter(key, value)
            else:
                module.register_buffer(key, value)
        else:
            module.register_buffer(key, value)


class BellMF(nn.Module):
    """
    Generalized bell shaped function.

    f(x) = 1 / 1 + abs((x - c) / a)^2b,

    where `a`, `b`, `c` and `d` are constants or learnable parameters.
    """

    def __init__(
            self,
            a: float = None,
            b: float = None,
            c: float = None,
            d: float = None,
            elementwise_affine: bool = None
    ):
        if a is None:
            a = 1.0
        if b is None:
            b = 1.0
        if c is None:
            c = 0.0
        if d is None:
            d = 1.0
        super().__init__()
        _register_function_params(self, elementwise_affine, a=a, b=b, c=c, d=d)

    def forward(self, x):
        return rf.bell(x, self.a, self.b, self.c, self.d)


class GaussianMF(nn.Module):
    """
    Gaussian function, the probability density function of the normal distribution.
    This is the archetypal bell shaped function and is frequently encountered in nature
    as a consequence of the central limit theorem.

    f(x) = a * exp(-(x - b)^2 / 2c^2),

    where `a`, `b`, `c` are constants or learnable parameters.
    """

    def __init__(
            self,
            a: float = None,
            b: float = None,
            c: float = None,
            elementwise_affine: bool = None
    ):
        if a is None:
            a = 1.0
        if b is None:
            b = 0.0
        if c is None:
            c = 1.0
        super().__init__()
        _register_function_params(self, elementwise_affine, a=a, b=b, c=c)

    def forward(self, x):
        return rf.gaussian(x, self.a, self.b, self.c)


class SechMF(nn.Module):
    """
    Hyperbolic secant. This is also the derivative of the Gudermannian function.

    f(x) = sech(x) = 2c / (a*e^x + b*e^-x),

    where `a`, `b`, `c` are constants or learnable parameters.
    """

    def __init__(
            self,
            a: float = None,
            b: float = None,
            c: float = None,
            elementwise_affine: bool = None
    ):
        if a is None:
            a = 0.0
        if b is None:
            b = 0.0
        if c is None:
            c = 1.0
        super().__init__()
        _register_function_params(self, elementwise_affine, a=a, b=b, c=c)

    def forward(self, x):
        return rf.sech(x, self.a, self.b, self.c)


class WoaMF(nn.Module):
    """
    Witch of Agnesi, the probability density function of the Cauchy distribution.
    This is also a scaled version of the derivative of the arcus tangent function.

    f(x) = f(x) = a * (4b^3 / (x^2 + 4b^2)),

    where `a` and `b` are constants or learnable parameters.
    """

    def __init__(
            self,
            a: float = None,
            b: float = None,
            elementwise_affine: bool = None
    ):
        if a is None:
            a = 1.0
        if b is None:
            b = 1.0
        super().__init__()
        _register_function_params(self, elementwise_affine, a=a, b=b)

    def forward(self, x):
        return rf.woa(x, self.a, self.b)


class BumpMF(nn.Module):
    """
    Bump function.

    f(x) = a * exp(b^2 / ((x+c)^2 - b^2) + 1), if abs(x+c)<b, 0 otherwise,

    where `a` is a constant or learnable parameter.
    """

    def __init__(
            self,
            a: float = None,
            b: float = None,
            c: float = None,
            elementwise_affine: bool = None
    ):
        if a is None:
            a = 1.0
        if b is None:
            b = 1.0
        if c is None:
            c = 0.0
        super().__init__()
        _register_function_params(self, elementwise_affine, a=a, b=b, c=c)

    def forward(self, x):
        return rf.bump(x, self.a, self.b, self.c)


class SigmoidMF(nn.Module):
    """
    Modified version of the logistic (sigmoid) function.

    f(x) = a * exp(x+c)^b / (1 + exp(x+c))^2,

    where `a`, `b` and `c` are constants or learnable parameters.
    """

    def __init__(
            self,
            a: float = None,
            b: float = None,
            c: float = None,
            elementwise_affine: bool = None
    ):
        if a is None:
            a = 1.0
        if b is None:
            b = 0.0
        if c is None:
            c = 0.0
        super().__init__()
        _register_function_params(self, elementwise_affine, a=a, b=b, c=c)

    def forward(self, x):
        return rf.sigmoid(x, self.a, self.b, self.c)


class TriangleMF(nn.Module):
    """
    Triangle function.

    f(x) =
        - 0, if x<=a;
        - (x-a)/(b-a), if a<=x<=b;
        - (c-x)/(c-b), if b<=x<=c;
        - 0 otherwise (c<=x),

    where a and b are constants or learnable parameters.
    """

    def __init__(
            self,
            a: float = None,
            b: float = None,
            c: float = None,
            d: float = None,
            elementwise_affine: bool = None
    ):
        if a is None:
            a = -1.0
        if b is None:
            b = 0.0
        if c is None:
            c = 1.0
        if d is None:
            d = 1.0
        a, b, c = sorted((a, b, c))
        super().__init__()
        _register_function_params(self, elementwise_affine, a=a, b=b, c=c, d=d)

    def forward(self, x):
        return rf.tri(x, self.a, self.b, self.c, self.d)


class TrapezoidalMF(nn.Module):
    """
    Trapezoidal function.

    f(x) =
        - 0, if x<=a;
        - (x-a)/(b-a), if a<=x<=b;
        - 1, if b<=x<=c;
        - (c-x)/(c-b), if c<=x<=d;
        - 0 otherwise (d<=x),

    where a and b are constants or learnable parameters.
    """

    def __init__(
            self,
            a: float = None,
            b: float = None,
            c: float = None,
            d: float = None,
            e: float = None,
            elementwise_affine: bool = None
    ):
        if a is None:
            a = -1.0
        if b is None:
            b = -0.5
        if c is None:
            c = 0.5
        if d is None:
            d = 1.0
        if e is None:
            e = 1.0
        a, b, c, d = sorted((a, b, c, d))
        super().__init__()
        _register_function_params(self, elementwise_affine, a=a, b=b, c=c, d=d, e=e)

    def forward(self, x):
        return rf.trap(x, self.a, self.b, self.c, self.d, self.e)
