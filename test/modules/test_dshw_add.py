import torch
from torch import nn

from torches.models.pipeline_model import PipelineModel
from torches.modules.learners.dshw_add import DSHWAdditiveLearner
from torches.modules.learners.dshw_offset_based_add import DSHWOffsetBasedAdditiveLearner


ROLLBY_FOR_TEST_1 = 2
ROLLBY_FOR_TEST_2 = 26


def gen_period_labels(period_len, cutby, rollby, ratio=0):
    a = torch.arange(0, period_len).repeat(cutby // period_len + 1 + ratio).view(1, -1, 1)
    a = torch.roll(a, rollby, dims=1)
    return a[:, :cutby, :]


def test_smoke_conv_learner(dataset_2_batch):

    loss_fn = nn.MSELoss()
    h = 24
    xh = dataset_2_batch.torch_ts.size(1) + h

    x0 = dataset_2_batch.torch_ts.view(2, -1, 1)

    # sequences must match in length before concatenating them.
    ratio = dataset_2_batch.period2 // dataset_2_batch.period1

    seas_mask = torch.cat([
        gen_period_labels(dataset_2_batch.period1, xh, ROLLBY_FOR_TEST_1, ratio=ratio).repeat(2, 1, 1),
        gen_period_labels(dataset_2_batch.period2, xh, ROLLBY_FOR_TEST_2).repeat(2, 1, 1)
    ], 2)

    exog_cat0 = torch.empty(2, xh, 1)
    exog_cnt0 = torch.empty(2, xh, 1)
    y0 = None

    # new impl
    model = PipelineModel([
        DSHWAdditiveLearner(period1=dataset_2_batch.period1, period2=dataset_2_batch.period2, h=h),
    ], loss_fn=loss_fn)

    # that's actually the old implementation
    model2 = PipelineModel([
        DSHWOffsetBasedAdditiveLearner(period1=dataset_2_batch.period1, period2=dataset_2_batch.period2, h=h),
    ], loss_fn=loss_fn)

    out = model.forward(x0, seas_mask, exog_cat0, exog_cnt0, y0)
    out2 = model2.forward(x0, seas_mask, exog_cat0, exog_cnt0, y0)

    assert torch.allclose(out.forecast, out2.forecast)
    out2.loss_result.overall.backward()
