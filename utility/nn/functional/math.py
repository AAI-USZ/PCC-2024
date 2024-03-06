import torch
from torch import Tensor

__all__ = (
    'bell',
    'gaussian',
    'sech',
    'woa',
    'bump',
    'sigmoid',
    'tri',
    'trap'
)


def bell(
        x,
        a: float | Tensor = None,
        b: float | Tensor = None,
        c: float | Tensor = None,
        d: float | Tensor = None
):
    """
    Generalized membership bell-shaped function.

    Args:
        x: inputs
        a: constant
        b: constant
        c: constant
        d: constant

    Returns:
        f(x) = d / (1 + abs((x - c) / a)^2b)
    """

    if a is None:
        a = 1.0
    if b is None:
        b = 1.0
    if c is None:
        c = 0.0
    if d is None:
        d = 1.0

    return d / (1 + torch.abs((torch.square(x - c) / a)) ** (2 * b))


def gaussian(
        x,
        a: float | Tensor = None,
        b: float | Tensor = None,
        c: float | Tensor = None
):
    """
    Gaussian function, the probability density function of the normal distribution.
    This is the archetypal bell shaped function and is frequently encountered in nature
    as a consequence of the central limit theorem.

    Args:
        x: inputs
        a: constant
        b: constant
        c: constant

    Returns:
        f(x) = a * exp(-(x-b)^2 / 2c^2)
    """

    if a is None:
        a = 1.0
    if b is None:
        b = 0.0
    if c is None:
        c = 1.0

    return a * torch.exp(-torch.square(x - b) / (2 * c ** 2))


def sech(
        x,
        a: float | Tensor = None,
        b: float | Tensor = None,
        c: float | Tensor = None
):
    """
    Hyperbolic secant. This is also the derivative of the Gudermannian function.

    Args:
        x: inputs
        a: constant
        b: constant
        c: constant

    Returns:
        f(x) = sech(x) = 2c / (e^(x+a) + e^-(x+b)
    """

    if a is None:
        a = 0.0
    if b is None:
        b = 0.0
    if c is None:
        c = 1.0

    return 2 * c / (torch.exp(x + a) + torch.exp(-x - b))


def woa(
        x,
        a: float | Tensor = None,
        b: float | Tensor = None
):
    """
    Witch of Agnesi, the probability density function of the Cauchy distribution.
    This is also a scaled version of the derivative of the arcus tangent function.

    Args:
        x: input
        a: constant
        b: constant

    Returns:
        f(x) = a * (4b^3 / (x^2 + 4b^2))
    """

    if a is None:
        a = 1.0
    if b is None:
        b = 1.0

    return a * (4 * b ** 3 / (torch.square(x) + 4 * b ** 2))


def bump(
        x,
        a: float | Tensor = None,
        b: float | Tensor = None,
        c: float | Tensor = None
):
    """
    Bump function.

    Args:
        x: inputs
        a: constant
        b: constant
        c: constant

    Returns:
        f(x) = a * exp(b^2 / ((x+c)^2 - b^2) + 1), if abs(x+c)<b, 0 otherwise
    """

    if a is None:
        a = 1.0
    if b is None:
        b = 1.0
    if c is None:
        c = 0.0

    y = x
    # noinspection PyTypeChecker
    mask: torch.Tensor = torch.abs(x + c) < b
    y[mask] = a * torch.exp(b ** 2 / (torch.square(x + c) - b ** 2) + 1)
    y[mask.logical_not()] = 0
    return y


def sigmoid(
        x,
        a: float | Tensor = None,
        b: float | Tensor = None,
        c: float | Tensor = None
):
    """
    Modified version of the logistic (sigmoid) function.

    Args:
        x: inputs
        a: constant
        b: constant
        c: constant

    Returns:
        f(x) = a * exp(x+c)^b / (1 + exp(x+c))^2
    """

    if a is None:
        a = 1.0
    if b is None:
        b = 0.0
    if c is None:
        c = 0.0

    return a * torch.exp(x + c) ** b / torch.square(1 + torch.exp(x + c))


def tri(
        x,
        a: float | Tensor = None,
        b: float | Tensor = None,
        c: float | Tensor = None,
        d: float | Tensor = None
):
    """
    Triangular function.

    Args:
        x: inputs
        a: constant
        b: constant
        c: constant
        d: constant

    Returns:
        f(x) =
        0, if x<=a;
        (x-a)/(b-a), if a<=x<=b;
        (c-x)/(c-b), if b<=x<=c;
        0 otherwise (c<=x)
    """

    if a is None:
        a = -1.0
    if b is None:
        b = 0.0
    if c is None:
        c = 1.0
    if d is None:
        d = 1.0
    a, b, c = sorted((a, b, c))

    return d * torch.maximum(
        torch.minimum(
            (x - a) / (b - a), (c - x) / (c - b)
        ),
        torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    )


def trap(
        x,
        a: float | Tensor = None,
        b: float | Tensor = None,
        c: float | Tensor = None,
        d: float | Tensor = None,
        e: float | Tensor = None
):
    """
    Trapezoidal function.

    Args:
        x: inputs
        a: constant
        b: constant
        c: constant
        d: constant
        e: constant

    Returns:
        f(x) =
        0, if x<=a;
        (x-a)/(b-a), if a<=x<=b;
        1, if b<=x<=c;
        (c-x)/(c-b), if c<=x<=d;
        0 otherwise (d<=x)
    """

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

    return e * torch.maximum(
        torch.minimum(
            torch.minimum(
                (x - a) / (b - a),
                (d - x) / (d - c)
            ),
            torch.ones(x.shape, dtype=x.dtype, device=x.device)
        ),
        torch.zeros(x.shape, dtype=x.dtype, device=x.device))
