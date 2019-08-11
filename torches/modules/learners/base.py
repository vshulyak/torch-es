import torch.nn as nn


class BaseStatefulContainer(object):
    """
    Base class for all StatefulContainers.

    StatefulContainers hold temporary state while iterating over the sequence.
    """
    def __init__(self, learner, x, seas_mask, exog_cat, exog_cnt):
        super().__init__()
        self.learner = learner
        self.x = x
        self.seas_mask = seas_mask
        self.exog_cat = exog_cat
        self.exog_cnt = exog_cnt

        self.bs, self.n, self.f = x.size()

    def init(self):
        pass

    def step(self, i, state):
        raise NotImplemented

    def get_uncertainty(self, forecast):
        return {
            'ih': None,
            'il': None
        }

    def get_history(self):
        return {}

    def get_losses(self, loss_fn):
        return {}

    def finalize(self):
        pass


class BaseLearner(nn.Module):
    """
    Base for all Learners.

    Learner classes hold Parameters
    """
    STATEFUL_CONTAINER_CLASS = BaseStatefulContainer

    def __init__(self):
        super().__init__()

    def build_stateful_container(self, x, seas_mask, exog_cat, exog_cnt):
        return self.STATEFUL_CONTAINER_CLASS(self, x, seas_mask, exog_cat, exog_cnt)
