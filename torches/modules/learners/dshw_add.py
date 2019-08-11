import torch
from torch import nn

from .base import BaseLearner, BaseStatefulContainer

from torches.utils.modelparams import pytorch_diff, siginv


DEFAULT_PARAM_VALUE = 1e-03


def add_init_level_trend(y, period1_dim, period2_dim, enable_trend):
    assert y.ndimension() == 3, "3-dim input is expected"

    bs, n, f = y.size()

    if enable_trend:
        x = torch.cat([torch.zeros((bs, 1, f)), pytorch_diff(y[:, :period2_dim, :])], dim=1)
        t = torch.mean(((y[:, :period2_dim, :] - y[:, period2_dim:(2 * period2_dim), :]) / period2_dim) + x, dim=1) / 2  # trend
        s = torch.mean(y[:, :(2 * period2_dim), :], dim=1) - (period2_dim + 0.5) * t  # level
    else:
        t = 0  # TODO: 1 if multiplicative
        s = torch.mean(y[:, :(2 * period2_dim), :], dim=1)

    return t, s


class HWStatefulContainer(BaseStatefulContainer):

    def __init__(self, learner, x, seas_mask, exog_cat, exog_cnt):
        super().__init__(learner, x, seas_mask, exog_cat, exog_cnt)

        self.yhat = torch.empty((self.bs, self.n, self.f))

        self.init_Ic = learner.init_Ic
        self.init_wc = learner.init_wc

        self.period1_dim = learner.period1_dim
        self.period2_dim = learner.period2_dim

        self.alphas = torch.sigmoid(learner.alphas)
        self.betas = torch.sigmoid(learner.betas)
        self.gammas = torch.sigmoid(learner.gammas)
        self.omegas = torch.sigmoid(learner.omegas)

    def init(self):

        self.Ic = self.init_Ic.view(1, -1, 1).repeat(self.bs, 1, 1)
        self.wc = self.init_wc.view(1, -1, 1).repeat(self.bs, 1, 1)

        self.t, self.s = add_init_level_trend(self.x, self.period1_dim, self.period2_dim, self.learner.enable_trend)

        self.t_history = []
        self.s_history = []

    def step(self, i, state):

        bi = torch.arange(0, self.bs)  # batch index, matching seasonal mask index. Single item selection.
        si1 = self.seas_mask[:, i, 0]
        si2 = self.seas_mask[:, i, 1]

        intervention_term = state['intervention_term'] if 'intervention_term' in state else 0  # TODO: mult vs add

        yh = (self.s + self.t) + self.Ic[bi, si1, :] + self.wc[bi, si2, :]
        self.yhat[:, i, :] = yh + intervention_term  # apply fix from the last iteration
        snew = self.alphas * (self.x[:, i, :] - (self.Ic[bi, si1, :] + self.wc[bi, si2, :] + intervention_term)) + (1 - self.alphas) * (self.s + self.t)
        tnew = self.betas * (snew - self.s) + (1 - self.betas) * self.t

        self.s = snew
        self.t = tnew if self.learner.enable_trend else torch.zeros(self.bs, 1)

        # these are inplace operations. In theory, gradients should fail for these. However, they don't for some reason in this case.
        self.Ic[bi, si1, :] = self.gammas * (self.x[:, i, :] - (snew + self.wc[bi, si2, :] + intervention_term)) + (1 - self.gammas) * self.Ic[bi, si1, :]
        self.wc[bi, si2, :] = self.omegas * (self.x[:, i, :] - (snew + self.Ic[bi, si1, :] + intervention_term)) + (1 - self.omegas) * self.wc[bi, si2, :]

        if not state:
            state = {}

        state.update({
            'es_s': self.s,
            'es_t': self.t,
            'es_yhat': self.yhat[:, i, :],
            'es_residual': self.x[:, i, :] - yh
        })

        self.t_history += [self.t]
        self.s_history += [self.s]
        return state

    def forecast(self, h):

        t = self.t
        s = self.s

        sm1 = self.seas_mask[:, self.n:, 0]
        sm2 = self.seas_mask[:, self.n:, 1]

        # batch index, matching seasonal mask index. Multiple item selection.
        bi = torch.arange(self.bs).view(-1, 1).repeat(1, sm1.size(1))

        level = s.view(self.bs, 1, self.f)
        trend = torch.arange(1, h + 1).float().view(1, -1, 1).repeat(self.bs, 1, self.f) * t.view(self.bs, 1, self.f)
        seas_1 = self.Ic[bi, sm1]
        seas_2 = self.wc[bi, sm2]

        return {
            'es': level + trend + seas_1 + seas_2
        }

    def get_losses(self, loss_fn):
        main_loss = {
            'es': loss_fn(self.x, self.yhat)
        }

        # adds smoothing loss for the seasonalities: penalizes too big differences.
        # another way to do this would be a smoothing spline penalty, via a second derivative (aka "double backprop"):
        #     grad = torch.autograd.grad(loss, w)[0]
        #     total_loss = loss + torch.norm(grad)
        # but this is slow to compute and a bit awkward in my tests
        if self.learner.enable_seas_smoothing:

            sh = torch.stack(self.s_history, 1)

            sl1 = torch.norm(self.init_Ic[1:] - self.init_Ic[:-1])
            sl2 = torch.norm(self.init_wc[1:] - self.init_wc[:-1])
            sl3 = torch.norm(sh[:, 1:, :] - sh[:, :-1, :]) / sh.size(1)
            return {**main_loss, **{
                'sm': sl1 + sl2 + sl3,
            }}
        return main_loss

    def get_history(self):
        return {
            'dshw_t_history': torch.stack(self.t_history, 1).detach(),
            'dshw_s_history': torch.stack(self.s_history, 1).detach()
        }


class DSHWAdditiveLearner(BaseLearner):
    """

    """
    STATEFUL_CONTAINER_CLASS = HWStatefulContainer

    def __init__(self, period1_dim, period2_dim, h,
                 enable_trend=True,
                 enable_seas_smoothing=True,
                 enable_hw_grad=True, enable_ar=False, enable_seas_grad=True,
                 out_name='intervention_term'):
        super().__init__()

        self.h = h
        self.period1_dim = period1_dim
        self.period2_dim = period2_dim
        self.enable_trend = enable_trend
        self.enable_seas_smoothing = enable_seas_smoothing
        self.enable_ar = enable_ar
        self.out_name = out_name

        self.alphas = nn.Parameter(siginv(torch.tensor([DEFAULT_PARAM_VALUE], requires_grad=enable_hw_grad)))
        self.betas = nn.Parameter(siginv(torch.tensor([DEFAULT_PARAM_VALUE], requires_grad=enable_hw_grad)))
        self.gammas = nn.Parameter(siginv(torch.tensor([DEFAULT_PARAM_VALUE], requires_grad=enable_hw_grad)))
        self.omegas = nn.Parameter(siginv(torch.tensor([DEFAULT_PARAM_VALUE], requires_grad=enable_hw_grad)))
        self.phis = nn.Parameter(siginv(torch.tensor([DEFAULT_PARAM_VALUE], requires_grad=enable_hw_grad)))

        # can be initialized later (optional)
        self.init_Ic = nn.Parameter(torch.zeros(period1_dim, requires_grad=enable_seas_grad))
        self.init_wc = nn.Parameter(torch.zeros(period2_dim, requires_grad=enable_seas_grad))
