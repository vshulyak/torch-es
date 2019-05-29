import numpy as np

import torch


def seasonality_ts(n, s_len=24):

    freq = n / s_len

    t = np.arange(n) / n
    c1 = 1.0 * np.sin(2 * np.pi * t * freq)
    c2 = .4 * np.sin(2 * np.pi * 15 * t)

    noise = np.random.rand(n)

    return c1 + c2 + noise


def trend(n, steepness=1.2):
    return np.arange(n) / (n**steepness) + np.random.rand(n) * 0.1


class DataSetOneFixture(object):

    n = 24 * 7 * 4 * 4
    h = 24 * 7 * 4
    level = 50
    period1 = 24
    period2 = 24 * 7

    @property
    def ts(self):
        np.random.seed(0)
        return (self.level +
                trend(self.n, steepness=0.9) +
                seasonality_ts(self.n, self.period1) +
                seasonality_ts(self.n, self.period2))

    @property
    def torch_ts(self):
        return torch.from_numpy(self.ts).float()


class DataSet2FeatureFixture(object):

    n = 24 * 7 * 4 * 4
    h = 24 * 7 * 4
    level1 = 50
    level2 = 150
    period1 = 24
    period2 = 24 * 7

    @property
    def ts(self):
        np.random.seed(1)
        a = (self.level1 +
             trend(self.n, steepness=0.9) +
             seasonality_ts(self.n, self.period1) +
             seasonality_ts(self.n, self.period2)).reshape(-1, 1)
        np.random.seed(0)
        b = (self.level2 +
             trend(self.n, steepness=1.0) +
             seasonality_ts(self.n, self.period1) +
             seasonality_ts(self.n, self.period2)).reshape(-1, 1)
        return np.concatenate([a, b], axis=1)

    @property
    def torch_ts(self):
        return torch.from_numpy(self.ts).float()


class DataSet2BatchFixture(object):

    n = 24 * 7 * 4 * 4
    h = 24 * 7 * 4
    level1 = 50
    level2 = 150
    period1 = 24
    period2 = 24 * 7

    @property
    def ts(self):
        np.random.seed(1)
        a = (self.level1 +
             trend(self.n, steepness=0.9) +
             seasonality_ts(self.n, self.period1) +
             seasonality_ts(self.n, self.period2)).reshape(1, -1)
        np.random.seed(0)
        b = (self.level2 +
             trend(self.n, steepness=1.0) +
             seasonality_ts(self.n, self.period1) +
             seasonality_ts(self.n, self.period2)).reshape(1, -1)
        return np.concatenate([a, b], axis=0)

    @property
    def torch_ts(self):
        return torch.from_numpy(self.ts).float()


class DataSet2Batch2FeatureFixture(object):

    n = 24 * 7 * 4 * 4 * 2
    h = 24 * 7 * 4
    level1 = 50
    level2 = 150
    period1 = 24
    period2 = 24 * 7

    @property
    def ts(self):
        np.random.seed(1)
        a = (self.level1 +
             trend(self.n, steepness=0.9) +
             seasonality_ts(self.n, self.period1) +
             seasonality_ts(self.n, self.period2)).reshape(2, -1, 1)
        np.random.seed(0)
        b = (self.level2 +
             trend(self.n, steepness=1.0) +
             seasonality_ts(self.n, self.period1) +
             seasonality_ts(self.n, self.period2)).reshape(2, -1, 1)
        return np.concatenate([a, b], axis=2)

    @property
    def torch_ts(self):
        return torch.from_numpy(self.ts).float()
