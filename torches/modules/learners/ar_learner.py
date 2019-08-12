import torch
from torch import nn

from .base import BaseLearner, BaseStatefulContainer

from torches.utils.modelparams import siginv


DEFAULT_PARAM_VALUE = 1e-03


class ARStatefulContainer(BaseStatefulContainer):

    def init(self):
        phis = torch.sigmoid(self.learner.phis)

        # trim too small values
        phis = torch.where(phis < 1e-7, torch.zeros(phis.size()), phis)
        # trim too big values
        phis = torch.where(phis > 0.9, torch.zeros(phis.size()), phis)

        self.phis = phis

        self.yhat = []

    def step(self, i, state):
        self.yhat += [state['es_yhat']]
        return state

    def finalize(self):
        self.yhat = torch.stack(self.yhat, 1)

        self.e = self.x - self.yhat

        self.yhat = self.yhat + self.phis * torch.cat([
            torch.zeros((self.bs, 1, self.f)),
            self.e[:, :-1, :]
        ], dim=1)

    def get_losses(self, loss_fn):
        return {
            'es_ar': loss_fn(self.x, self.yhat)
        }

    def forecast(self, h):
        powers = torch.arange(1, h + 1).float().view(1, -1, 1).repeat(self.bs, 1, self.f)
        error = self.e[:, self.n - 1, :].view(self.bs, 1, self.f)
        ar_forecast = self.phis ** powers * error
        return {
            'ar': ar_forecast
        }


class ARLearner(BaseLearner):

    STATEFUL_CONTAINER_CLASS = ARStatefulContainer

    def __init__(self):
        super().__init__()
        self.phis = nn.Parameter(siginv(torch.tensor([DEFAULT_PARAM_VALUE], requires_grad=True)))
