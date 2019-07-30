import functools
import itertools
import operator

import torch.nn as nn

from torches.utils.containers import LossResult, make_obj


class PipelineModel(nn.Module):
    """
    A Module which iterates over the provided submodules.
    """
    def __init__(self, pipeline, loss_fn, is_multiplicative=False, h=24):
        super().__init__()

        self.loss_fn = loss_fn
        self.h = h
        self.forecast_concat_op = operator.mul if is_multiplicative else operator.add

        self.pipeline = nn.ModuleList(pipeline)

    def forward(self, x, seas_mask, exog_cat, exog_cnt, y=None):

        # build
        stateful_containers = [
            p.build_stateful_container(x, seas_mask, exog_cat, exog_cnt)
            for p in self.pipeline
        ]

        # init
        [sc.init() for sc in stateful_containers]

        # step
        for i in range(x.size(1)):
            state = None  # TODO: Check if this influenced the result (I moved it inside the loop)
            for sc in stateful_containers:
                state = sc.step(i, state)
        del state  # needed?

        # merge forecasts from all models with a defined op (addition or multiplication)
        forecast = functools.reduce(self.forecast_concat_op, [sc.forecast(self.h) for sc in stateful_containers])

        # finalize conainers
        [sc.finalize() for sc in stateful_containers]

        # get losses from all submodels and merge them
        losses = dict(itertools.chain(*[sc.get_losses(self.loss_fn).items() for sc in stateful_containers]))

        # get histories to be added to the returned result
        histories = dict(itertools.chain(*[sc.get_history().items() for sc in stateful_containers]))

        del stateful_containers  # needed?

        # overall loss for learning
        overall = sum(losses.values())

        # aux losses for inspection and debugging.
        # the losses are detached here.
        losses = {k: v.item() for k, v in losses.items()}

        # forecast loss is computed only if 'y' is provided (ugly?)
        if y is not None:
            losses.update({
                'fc': self.loss_fn(forecast.detach(), y.detach())
            })

        loss_result = LossResult(overall=overall, losses=losses)

        return make_obj(loss_result=loss_result,
                        forecast=forecast,
                        histories=histories)
