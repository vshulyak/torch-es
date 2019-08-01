import torch
from torch import nn

from .base import BaseLearner, BaseStatefulContainer

from torches.utils.modelparams import pytorch_diff, siginv


DEFAULT_PARAM_VALUE = 1e-03


def add_init_level_trend(y, period1, period2):
    assert y.ndimension() == 3, "3-dim input is expected"

    bs, n, f = y.size()

    x = torch.cat([torch.zeros((bs, 1, f)), pytorch_diff(y[:, :period2, :])], dim=1)
    t = torch.mean(((y[:, :period2, :] - y[:, period2:(2 * period2), :]) / period2) + x, dim=1) / 2  # trend
    s = torch.mean(y[:, :(2 * period2), :], dim=1) - (period2 + 0.5) * t  # level

    return t, s


class HWStatefulContainer(BaseStatefulContainer):

    def __init__(self, learner, x, seas_mask, exog_cat, exog_cnt):
        super().__init__(learner, x, seas_mask, exog_cat, exog_cnt)

        self.yhat = torch.empty((self.bs, self.n, self.f))

        self.init_Ic = learner.init_Ic
        self.init_wc = learner.init_wc

        self.period1 = learner.period1
        self.period2 = learner.period2

        self.alphas = torch.sigmoid(learner.alphas)
        self.betas = torch.sigmoid(learner.betas)
        self.gammas = torch.sigmoid(learner.gammas)
        self.omegas = torch.sigmoid(learner.omegas)

    def init(self):

        period1_mask = self.seas_mask[:, :self.period1, 0]
        period2_mask = self.seas_mask[:, :self.period2, 1]

        Ic = []
        wc = []
        for i in range(self.bs):
            Ic += [self.init_Ic[period1_mask[i, :]]]
            wc += [self.init_wc[period2_mask[i, :]]]

        self.Ic = torch.stack(Ic, 0).unsqueeze(2)
        self.wc = torch.stack(wc, 0).unsqueeze(2)

        self.t, self.s = add_init_level_trend(self.x, self.period1, self.period2)

        self.t_history = []
        self.s_history = []

    def step(self, i, state):

        residual_pred = state['residual_pred'] if state and 'residual_pred' in state else 0  # TODO: mult vs add

        yh = (self.s + self.t) + self.Ic[:, i % self.period1, :] + self.wc[:, i % self.period2, :]
        self.yhat[:, i, :] = yh + residual_pred  # apply fix from the last iteration
        snew = self.alphas * (self.x[:, i, :] - (self.Ic[:, i % self.period1, :] + self.wc[:, i % self.period2, :] + residual_pred)) + (1 - self.alphas) * (self.s + self.t)
        tnew = self.betas * (snew - self.s) + (1 - self.betas) * self.t

        Ico = self.Ic.clone()
        wco = self.wc.clone()

        Ico[:, i % self.period1, :] = self.gammas * (self.x[:, i, :] - (snew + self.wc[:, i % self.period2, :] + residual_pred)) + (1 - self.gammas) * self.Ic[:, i % self.period1, :]
        wco[:, i % self.period2, :] = self.omegas * (self.x[:, i, :] - (snew + self.Ic[:, i % self.period1, :] + residual_pred)) + (1 - self.omegas) * self.wc[:, i % self.period2, :]
        self.Ic = Ico
        self.wc = wco

        self.s = snew
        self.t = tnew

        if not state:
            state = {}

        state.update({
            's': self.s,
            't': self.t,
            'residual': self.x[:, i, :] - yh
        })

        self.t_history += [self.t.unsqueeze(1)]
        self.s_history += [self.s.unsqueeze(1)]
        return state

    def forecast(self, h):
        Ic = torch.roll(self.Ic, -self.n % self.period1, dims=1)
        wc = torch.roll(self.wc, -self.n % self.period2, dims=1)

        t = self.t
        s = self.s

        ca = s.view(self.bs, 1, self.f) + torch.arange(1, h + 1).float().view(1, -1, 1).repeat(self.bs, 1, self.f) * t.view(self.bs, 1, self.f)
        cb = Ic.repeat(1, h // self.period1 + 1, 1)[:, :h, :]
        cc = wc.repeat(1, h // self.period2 + 1, 1)[:, :h, :]

        return ca + cb + cc

    def get_losses(self, loss_fn):
        return {
            'es': loss_fn(self.x, self.yhat)
        }

    def get_history(self):
        return {
            't_history': torch.stack(self.t_history, 1).squeeze(3).detach(),  # why do I get 4 dim?
            's_history': torch.stack(self.s_history, 1).squeeze(3).detach()
        }


class DSHWAdditiveLearner(BaseLearner):
    """

    TODO:
    - version without trend
    """
    STATEFUL_CONTAINER_CLASS = HWStatefulContainer

    def __init__(self, period1, period2, h,
                 enable_hw_grad=True, enable_ar=False, enable_seas_grad=True):
        super().__init__()

        self.h = h
        self.period1 = period1
        self.period2 = period2
        self.enable_ar = enable_ar

        self.alphas = nn.Parameter(siginv(torch.tensor([DEFAULT_PARAM_VALUE], requires_grad=enable_hw_grad)))
        self.betas = nn.Parameter(siginv(torch.tensor([DEFAULT_PARAM_VALUE], requires_grad=enable_hw_grad)))
        self.gammas = nn.Parameter(siginv(torch.tensor([DEFAULT_PARAM_VALUE], requires_grad=enable_hw_grad)))
        self.omegas = nn.Parameter(siginv(torch.tensor([DEFAULT_PARAM_VALUE], requires_grad=enable_hw_grad)))
        self.phis = nn.Parameter(siginv(torch.tensor([DEFAULT_PARAM_VALUE], requires_grad=enable_hw_grad)))

        # can be initialized later (optional)
        self.init_Ic = nn.Parameter(torch.zeros(period1, requires_grad=enable_seas_grad))
        self.init_wc = nn.Parameter(torch.zeros(period2, requires_grad=enable_seas_grad))
