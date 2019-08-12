import torch
from torch import nn

from .base import BaseLearner, BaseStatefulContainer


class PredictionQuantileStatefulContainer(BaseStatefulContainer):

    def init(self):
        self.yhat = []

    def step(self, i, state):
        self.yhat += [state['es_yhat']]
        return state

    def finalize(self):
        self.yhat = torch.stack(self.yhat, 1)
        self.lower_b = self.learner.lower(self.yhat)
        self.upper_b = self.learner.upper(self.yhat)

    def forecast(self, h):
        return {}

    def get_uncertainty(self, forecast):
        return {
            'il': self.learner.lower(forecast),
            'ih': self.learner.upper(forecast)
        }

    def get_losses(self, loss_fn):
        # TODO: MSIS?
        errors_l = self.x - self.lower_b
        errors_h = self.x - self.upper_b
        ql = 0.05
        qh = 0.95
        return {
            'il': torch.max((ql - 1) * errors_l, ql * errors_l).mean(),
            'ih': torch.max((qh - 1) * errors_h, qh * errors_h).mean()
        }


class PredictionQuantileLearner(BaseLearner):
    """
    Scales main forecast to approximate lower and upper bounds (for example, 0.05 and 0.95 quantiles).
    It would be logical to use MAE loss for the point forecasts if using this learner (so that you get quantiles
    everywhere).

    It would be much better, if the FCN here could access some kind of autoencoded state to provide something more
    than just a scaled forecast. However, this learner should be a good starting point for estimating quantiles.
    """
    STATEFUL_CONTAINER_CLASS = PredictionQuantileStatefulContainer

    def __init__(self):
        super().__init__()
        self.lower = nn.Linear(1, 1)
        self.upper = nn.Linear(1, 1)
