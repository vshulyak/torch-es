import functools
import itertools
import operator

import torch.nn as nn

from torches.utils.containers import LossResult, make_obj

filt_none = functools.partial(filter, None.__ne__)


class DefaultMerger(object):

    def __init__(self, h, forecast_concat_op, stateful_containers):
        self.h = h
        self.forecast_concat_op = forecast_concat_op
        self.stateful_containers = stateful_containers

    def merge_forecasts(self):
        # merge forecasts from all models with a defined op (addition or multiplication)
        return functools.reduce(self.forecast_concat_op,
                                itertools.chain(*[sc.forecast(self.h).values() for sc in self.stateful_containers]))

    def merge_histories(self):
        # get histories to be added to the returned result
        return dict(itertools.chain(*[sc.get_history().items() for sc in self.stateful_containers]))

    def merge_uncertainties(self, forecast):
        return {
            'ih': functools.reduce(self.forecast_concat_op,
                                   filt_none([sc.get_uncertainty(forecast)['ih'] for sc in self.stateful_containers]),
                                   0),  # TODO: 1 for mult
            'il': functools.reduce(self.forecast_concat_op,
                                   filt_none([sc.get_uncertainty(forecast)['il'] for sc in self.stateful_containers]),
                                   0)  # TODO: 1 for mult
        }

    def merge_losses(self, loss_fn, y_true, y_pred):

        # get losses from all submodels and merge them
        losses = dict(itertools.chain(*[sc.get_losses(loss_fn).items() for sc in self.stateful_containers]))

        # overall loss for learning
        overall = sum(losses.values())

        # aux losses for inspection and debugging.
        # the losses are detached here.
        losses = {k: v.item() for k, v in losses.items()}

        # forecast loss is computed only if 'y' is provided (ugly?)
        if y_true is not None:
            losses.update({
                'fc': loss_fn(y_pred.detach(), y_true.detach())
            })

        loss_result = LossResult(overall=overall, losses=losses)

        return loss_result


class PipelineModel(nn.Module):
    """
    A Module which iterates over the provided submodules.
    """
    def __init__(self, pipeline, loss_fn, is_multiplicative=False, h=24, merger_class=DefaultMerger):
        super().__init__()

        self.loss_fn = loss_fn
        self.h = h
        self.forecast_concat_op = operator.mul if is_multiplicative else operator.add

        self.pipeline = nn.ModuleList(pipeline)
        self.merger_class = merger_class

    def forward(self, x, seas_mask, exog_cat, exog_cnt, y=None):

        # build
        stateful_containers = [
            p.build_stateful_container(x, seas_mask, exog_cat, exog_cnt)
            for p in self.pipeline
        ]

        # init
        [sc.init() for sc in stateful_containers]

        # state is initialized at the beginning of each sequence. Then the result from i step is carried to i+1 step.
        state = {}

        # step
        for i in range(x.size(1)):
            for sc in stateful_containers:
                state = sc.step(i, state)
        del state  # needed?

        # finalize conainers
        [sc.finalize() for sc in stateful_containers]

        # merge all the information and create results
        merger = self.merger_class(self.h, self.forecast_concat_op, stateful_containers)
        forecast = merger.merge_forecasts()
        loss_result = merger.merge_losses(self.loss_fn, y, forecast)
        uncertainty = merger.merge_uncertainties(forecast)
        histories = merger.merge_histories()

        del stateful_containers  # needed?

        return make_obj(loss_result=loss_result,
                        uncertainty=uncertainty,
                        forecast=forecast,
                        histories=histories)
