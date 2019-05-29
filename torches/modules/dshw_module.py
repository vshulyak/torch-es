import torch
import torch.nn as nn

from torches.utils.modelparams import default_param, pytorch_diff
from torches.utils.seasonality import add_seasindex, mult_seasindex


DEFAULT_PARAM_VALUE = 1e-03


make_obj = lambda **kwargs: type('Res', (object,), kwargs)()


class DSHWModule(nn.Module):
    """
    Double Sesonal Holt Winters.

    This module provides double seasonal exponential smoothing for 3d data: <batch, series_len, series_index>.
    In other words, you can learn HW-coefficients for multiple timeseries at once with batching support.
    Losses for each series can be arbitrary.

    Reference implementation can be found at https://github.com/robjhyndman/forecast/blob/v8.7/R/dshw.r.

    WARN: max_h is used to compute forecast horizons. You have to cut it to :h for loss computation
    """
    # TODO: it probably doesn't make sense to have multiple "hs", since we can just predict for the biggest
    # and then cut the predictions in the loss function
    def __init__(self, input_dim, period1, period2, hs,
                 alphas=None, betas=None, gammas=None, omegas=None, phis=None,
                 enable_ar=False, is_multiplicative=True):
        super().__init__()
        assert len(hs) == input_dim

        # TODO: alphas=False => requires_grad=False

        self.input_dim = input_dim
        self.period1 = period1
        self.period2 = period2
        self.hs = torch.tensor(hs, requires_grad=False)
        self.enable_ar = enable_ar
        self.is_multiplicative = is_multiplicative

        alphas = default_param(input_dim, alphas, DEFAULT_PARAM_VALUE)
        betas = default_param(input_dim, betas, DEFAULT_PARAM_VALUE)
        gammas = default_param(input_dim, gammas, DEFAULT_PARAM_VALUE)
        omegas = default_param(input_dim, omegas, DEFAULT_PARAM_VALUE)
        phis = default_param(input_dim, phis, DEFAULT_PARAM_VALUE)

        self.alphas = torch.nn.Parameter(alphas)
        self.betas = torch.nn.Parameter(betas)
        self.gammas = torch.nn.Parameter(gammas)
        self.omegas = torch.nn.Parameter(omegas)
        self.phis = torch.nn.Parameter(phis)

    def mult_init_params(self, y):
        assert y.ndimension() == 3, "3-dim input is expected"

        bs, n, f = y.size()

        # Starting values
        I1 = mult_seasindex(y, self.period1)
        w1 = mult_seasindex(y, self.period2)

        ratio = self.period2 // self.period1
        w1 = w1 / I1.repeat(1, ratio, 1)

        # TODO: torch.pad
        x = torch.cat([torch.zeros((bs, 1, f)), pytorch_diff(y[:, :self.period2, :])], dim=1)
        t = torch.mean(((y[:, :self.period2, :] - y[:, self.period2:(2 * self.period2), :]) / self.period2) + x, dim=1) / 2  # trend
        s = torch.mean(y[:, :(2 * self.period2), :], dim=1) - (self.period2 + 0.5) * t  # level

        return I1, w1, t, s

    def add_init_params(self, y):
        assert y.ndimension() == 3, "3-dim input is expected"

        bs, n, f = y.size()

        # Starting values
        I1 = add_seasindex(y, self.period1)
        w1 = add_seasindex(y, self.period2)

        ratio = self.period2 // self.period1
        w1 = w1 - I1.repeat(1, ratio, 1)

        x = torch.cat([torch.zeros((bs, 1, f)), pytorch_diff(y[:, :self.period2, :])], dim=1)
        t = torch.mean(((y[:, :self.period2, :] - y[:, self.period2:(2 * self.period2), :]) / self.period2) + x, dim=1) / 2  # trend
        s = torch.mean(y[:, :(2 * self.period2), :], dim=1) - (self.period2 + 0.5) * t  # level

        return I1, w1, t, s

    def get_coeffs(self):
        return {c: torch.sigmoid(getattr(self, c)) for c in ['alphas', 'betas', 'gammas', 'omegas', 'phis']}

    def add_update(self, y, Ic, wc, s, t):

        bs, n, f = y.size()

        alphas = torch.sigmoid(self.alphas)
        betas = torch.sigmoid(self.betas)
        gammas = torch.sigmoid(self.gammas)
        omegas = torch.sigmoid(self.omegas)

        yhat = torch.empty((bs, n, f))

        for i in range(n):
            yhat[:, i, :] = (s + t) + Ic[:, i % self.period1, :] + wc[:, i % self.period2, :]
            snew = alphas * (y[:, i, :] - (Ic[:, i % self.period1, :] + wc[:, i % self.period2, :])) + (1 - alphas) * (s + t)
            tnew = betas * (snew - s) + (1 - betas) * t

            Ico = Ic.clone()
            wco = wc.clone()

            Ico[:, i % self.period1, :] = gammas * (y[:, i, :] - (snew - wc[:, i % self.period2, :])) + (1 - gammas) * Ic[:, i % self.period1, :]
            wco[:, i % self.period2, :] = omegas * (y[:, i, :] - (snew - Ic[:, i % self.period1, :])) + (1 - omegas) * wc[:, i % self.period2, :]
            Ic = Ico
            wc = wco

            s = snew
            t = tnew

        return yhat, Ic, wc, s, t

    def mult_update(self, y, Ic, wc, s, t):

        bs, n, f = y.size()

        alphas = torch.sigmoid(self.alphas)
        betas = torch.sigmoid(self.betas)
        gammas = torch.sigmoid(self.gammas)
        omegas = torch.sigmoid(self.omegas)

        yhat = torch.empty((bs, n, f))

        for i in range(n):
            yhat[:, i, :] = (s + t) * Ic[:, i % self.period1, :] * wc[:, i % self.period2, :]
            snew = alphas * (y[:, i, :] / (Ic[:, i % self.period1, :] * wc[:, i % self.period2, :])) + (1 - alphas) * (s + t)
            tnew = betas * (snew - s) + (1 - betas) * t

            Ico = Ic.clone()
            wco = wc.clone()

            Ico[:, i % self.period1, :] = gammas * (y[:, i, :] / (snew * wc[:, i % self.period2, :])) + (1 - gammas) * Ic[:, i % self.period1, :]
            wco[:, i % self.period2, :] = omegas * (y[:, i, :] / (snew * Ic[:, i % self.period1, :])) + (1 - omegas) * wc[:, i % self.period2, :]
            Ic = Ico
            wc = wco

            s = snew
            t = tnew

        return yhat, Ic, wc, s, t

    def forward(self, y, state=None):
        # We need 3 dimentions: (batch,series_len,feature)
        assert y.ndimension() == 3, "3-dim input is expected"
        assert y.size(1) >= 2 * self.period2, "Input series too short"

        if not state:
            if self.is_multiplicative:
                state = self.mult_init_params(y)
            else:
                state = self.add_init_params(y)
        Ic, wc, t, s = state

        phis = torch.sigmoid(self.phis)

        # Allocate space
        bs, n, f = y.size()
        e = torch.empty((bs, n, f))
        max_h = self.hs.max()

        if self.is_multiplicative:
            yhat, Ic, wc, s, t = self.mult_update(y, Ic, wc, s, t)
        else:
            yhat, Ic, wc, s, t = self.add_update(y, Ic, wc, s, t)

        # we need to shift seasonality as it doesn't match the last point when we finished updating it.
        # this happens when n%period!=0, so we have to roll by this amount, shifting last elements to be the
        # first ones
        Ic = torch.roll(Ic, -n % self.period1, dims=1)
        wc = torch.roll(wc, -n % self.period2, dims=1)

        e = y - yhat

        # do not compute fcast if we're not asked for it (h=0), we can do only yhat fit
        if max_h > 0:

            ca = s.view(bs, 1, f) + torch.arange(1, max_h + 1).float().view(1, -1, 1).repeat(bs, 1, f) * t.view(bs, 1, f)
            cb = Ic.repeat(1, max_h // self.period1 + 1, 1)[:, :max_h, :]
            cc = wc.repeat(1, max_h // self.period2 + 1, 1)[:, :max_h, :]

            if self.is_multiplicative:
                fcast = ca * cb * cc
            else:
                fcast = ca + cb + cc

            # TODO: additive
            if self.enable_ar:
                # trim too small values
                phis = torch.where(phis < 1e-7, torch.zeros(phis.size()), phis)
                # trim too big values
                phis = torch.where(phis > 0.9, torch.zeros(phis.size()), phis)

                # applied only if phi is bigger than 0, otherwise the result stays the same
                yhat = yhat + phis * torch.cat([torch.zeros((bs, 1, f)), e[:, :-1, :]], dim=1)
                fcast = fcast + phis ** torch.arange(1, max_h + 1).float().view(1, -1, 1).repeat(bs, 1, f) * e[:, n - 1, :].view(bs, 1, f)
                e = y - yhat
        else:
            fcast = torch.empty((bs, max_h, f))

        return make_obj(mean=fcast,
                        yhat=yhat,
                        state=(Ic, wc, t, s),
                        e=e)

    def get_loss(self, x, y, loss_fn, state=None):

        ret = self.forward(x, state=state)

        mse = torch.mean(ret.e[:, :, 0]**2)

        mseh = loss_fn(ret.mean[:, :, 1], y[:, :, 1])

        rloss = (mse + mseh)

        return ret.mean, rloss
