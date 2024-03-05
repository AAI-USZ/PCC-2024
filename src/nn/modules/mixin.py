import numpy as np
import torch
from sklearn.base import RegressorMixin
from torch import nn


class TorchRegressor(nn.Module):
    def __init__(self, regressor: RegressorMixin):
        super().__init__()
        assert hasattr(regressor, 'predict')
        self.regressor = regressor

    def forward(self, x):
        device = x.device
        with torch.no_grad():
            x = x.cpu().numpy()
            # noinspection PyUnresolvedReferences
            y = self.regressor.predict(x).astype(np.float32)
            y = torch.from_numpy(y)
            y = y.to(device)
        return y

    def __str__(self):
        return f'{self.__class__.__name__}({str(self.regressor)})'

    def __repr__(self):
        return str(self)
