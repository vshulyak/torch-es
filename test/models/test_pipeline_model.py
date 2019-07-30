import torch
from torch import nn

from torches.models.pipeline_model import PipelineModel
from torches.modules.learners.dshw_add import DSHWAdditiveLearner
from torches.modules.learners.exog_conv import ExogConvLearner
from torches.modules.learners.one_step_lstm import OneStepLSTMLearner


def test_smoke_conv_learner(dataset_one):

    loss_fn = nn.MSELoss()

    h = 24
    xh = dataset_one.torch_ts.size(0) + h

    model = PipelineModel([
        DSHWAdditiveLearner(period1=dataset_one.period1, period2=dataset_one.period2, h=h),
        ExogConvLearner(cnt_input_dim=1)
    ], loss_fn=loss_fn)

    x0 = dataset_one.torch_ts.view(1, -1, 1)
    seas_mask = torch.cat([
        torch.arange(0, dataset_one.period1).repeat(xh + 1)[:xh].view(1, -1, 1),
        torch.arange(0, dataset_one.period2).repeat(xh + 1)[:xh].view(1, -1, 1)
    ], 2)
    exog_cat0 = torch.empty(1, xh, 1)
    exog_cnt0 = torch.empty(1, xh, 1)
    y0 = None

    out = model.forward(x0, seas_mask, exog_cat0, exog_cnt0, y0)
    out.loss_result.overall.backward()


def test_smoke_one_step_lstm_learner(dataset_one):

    loss_fn = nn.MSELoss()

    h = 24
    xh = dataset_one.torch_ts.size(0) + h

    model = PipelineModel([
        DSHWAdditiveLearner(period1=dataset_one.period1, period2=dataset_one.period2, h=h),
        OneStepLSTMLearner(cnt_input_dim=1)
    ], loss_fn=loss_fn)

    x0 = dataset_one.torch_ts.view(1, -1, 1)
    seas_mask = torch.cat([
        torch.arange(0, dataset_one.period1).repeat(xh + 1)[:xh].view(1, -1, 1),
        torch.arange(0, dataset_one.period2).repeat(xh + 1)[:xh].view(1, -1, 1)
    ], 2)
    exog_cat0 = torch.empty(1, xh, 1)
    exog_cnt0 = torch.empty(1, xh, 1)
    y0 = None

    out = model.forward(x0, seas_mask, exog_cat0, exog_cnt0, y0)
    out.loss_result.overall.backward()
