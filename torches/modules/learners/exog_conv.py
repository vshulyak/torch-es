from torch import nn

from .base import BaseLearner, BaseStatefulContainer


class ExogConvLearnerStatefulContainer(BaseStatefulContainer):
    """
    """
    def init(self):
        # pass through the series in one shot. Dims are <bs x seq_len x feature_len>
        self.residual_preds = self.learner.conv(self.exog_cnt.permute(0, 2, 1)).permute(0, 2, 1)

    def step(self, i, state):
        state.update({
            'residual_pred': self.residual_preds[:, i, :]
        })
        return state

    def forecast(self, h):
        return self.residual_preds[:, self.n:, :]


class ExogConvLearner(BaseLearner):
    """
    Runs a convolution op over the continuous exog variables to learn their effect on the dependent variable.
    """
    STATEFUL_CONTAINER_CLASS = ExogConvLearnerStatefulContainer

    def __init__(self, cnt_input_dim, kernel_size=3):
        super().__init__()

        self.conv = nn.Conv1d(cnt_input_dim, 1,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=1,
                              dilation=1,
                              padding_mode='zeros',
                              bias=False)
