import torch
from torch import nn

from torches.modules.misc.gaussian_noise import GaussianNoise
from torches.modules.misc.embedding_net import EmbeddingNet

from .base import BaseLearner, BaseStatefulContainer


class OneStepLSTMStatefulContainer(BaseStatefulContainer):

    def init(self):
        self.hidden1 = self.learner.init_hiddens(self.bs)
        self.hidden2 = self.learner.init_hiddens(self.bs)

        self.ins = []
        self.outs = []

    def step(self, i, state):

        cur_residual = state['residual']

        out, self.hidden1, self.hidden2 = self.learner(cur_residual,
                                                       self.exog_cat[:, i, :], self.exog_cnt[:, i, :],
                                                       self.hidden1, self.hidden2)

        self.ins += [cur_residual]
        self.outs += [out]

        state.update({
            'residual_pred': out
        })

        return state

    def get_losses(self, loss_fn):

        ins = torch.stack(self.ins, 1)
        outs = torch.stack(self.outs, 1)

        lm = loss_fn(outs[:, :-1, :], ins[:, 1:, 0:1])

        return {
            'lm': lm,
        }

    def forecast(self, h):

        fc = []

        hidden1_, hidden2_ = self.hidden1, self.hidden2

        out = self.outs[-1]

        exog_cat = self.exog_cat[:, self.n:, :]
        exog_cnt = self.exog_cnt[:, self.n:, :]

        for i in range(h):
            cat_t = exog_cat[:, i, :]
            cnt_t = exog_cnt[:, i, :]
            out, hidden1_, hidden2_ = self.learner(out, cat_t, cnt_t, hidden1_, hidden2_)
            fc += [out]

        fc = torch.stack(fc, 1)

        return fc


class OneStepLSTMLearner(BaseLearner):
    """
    A simple dummy 1 step LSTM learner. Probably ineffective as it is without a multistep loss, unless you
    have some good correlating exog features.
    """

    STATEFUL_CONTAINER_CLASS = OneStepLSTMStatefulContainer

    def __init__(self, cnt_input_dim,
                 cat_input_emb_conf=[],
                 input_noise=0, input_dropout=0, input_hidden_dim=4,
                 exog_cat_noise=0, exog_cat_dropout=0,
                 exog_cnt_noise=0, exog_cnt_dropout=0):
        super().__init__()

        x_dim = 1

        self.input_hidden_dim = input_hidden_dim

        self.enable_cnt = cnt_input_dim > 0
        self.enable_cat = len(cat_input_emb_conf) > 0
        cat_emb_out_f_n = sum([c[1] for c in cat_input_emb_conf])

        self.input_dropout = nn.Dropout(input_dropout)
        self.input_noise = GaussianNoise(input_noise)
        self.cnt_dropout = nn.Dropout(exog_cnt_noise)
        self.cnt_noise = GaussianNoise(exog_cnt_noise)
        self.cat_dropout = nn.Dropout(exog_cat_noise)
        self.cat_noise = GaussianNoise(exog_cat_noise)

        # cat_input_emb_conf example: [(7,3),(24,4)]
        self.cat_emb = EmbeddingNet(cat_input_emb_conf)

        self.lstm1 = nn.LSTMCell(x_dim + cnt_input_dim + cat_emb_out_f_n, self.input_hidden_dim, bias=False)
        self.lstm2 = nn.LSTMCell(self.input_hidden_dim, self.input_hidden_dim, bias=False)
        self.linear1 = nn.Linear(self.input_hidden_dim, self.input_hidden_dim, bias=False)
        self.linear2 = nn.Linear(self.input_hidden_dim, x_dim, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, x, exog_cat, exog_cnt, hidden1, hidden2):

        inputs = filter(None.__ne__, [
            # main "x" input
            self.input_dropout(self.input_noise(x)),

            # continuous inputs
            self.cnt_dropout(self.cnt_noise(exog_cnt)) if self.enable_cnt else None,

            # categorical inputs
            self.cat_dropout(self.cat_noise(self.cat_emb(exog_cnt))) if self.enable_cat else None
        ])
        x = torch.cat(list(inputs), dim=1)

        h_t, c_t = self.lstm1(x, hidden1)
        h_t2, c_t2 = self.lstm2(h_t, hidden2)

        hidden1 = h_t, c_t
        hidden2 = h_t2, c_t2

        output = h_t2
        output = self.linear1(output)
        output = self.tanh(output)  # relu was screwing the loss for some reason
        output = self.linear2(output)

        # For multiplicative: we want to scale the existing forecast by a coefficient in range (0,1).
        # so we can use tanh+1 or sigmoid+0.5
#         output = self.relu(output) #+ 0.0001
#         output = self.tanh(output) + 1
#         output = torch.sigmoid(output) + 0.5

        # For additive: we don't really care, but we can limit the additive effect.
        # The problem then is that aux loss might cause this value to have a huge bias...
        # output = torch.sigmoid(output) * 5 # is it a good idea?

        return output, hidden1, hidden2

    def init_hiddens(self, batch_size):
        h_t = torch.zeros(batch_size, self.input_hidden_dim, dtype=torch.float)
        c_t = torch.zeros(batch_size, self.input_hidden_dim, dtype=torch.float)
        return h_t, c_t
