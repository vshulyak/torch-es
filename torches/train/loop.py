from collections import defaultdict, OrderedDict

import torch

from tqdm import tqdm_notebook as tqdm


def fmt_losses(losses):
    overall = sum(losses.values())
    return f'Loss: {overall:.4f}, ' + ', '.join([f'{k}: {v:.4f}' for k, v in losses.items()])


class ProgressBar(object):
    """
    Progress bar for pytorch, with tqdm support.

    Inspired by this project: https://github.com/devforfu/loop
    """
    def __init__(self):
        self.init_counters()

    def init_counters(self):
        self.values = defaultdict(lambda: defaultdict(int))
        self.counts = defaultdict(lambda: defaultdict(int))

    def training_started(self, phases, **kwargs):
        bars = OrderedDict()
        for phase in phases:
            bars[phase.name] = tqdm(total=len(phase.dataloader), desc=phase.name)
        self.bars = bars

    def batch_ended(self, phase, losses, target_size, **kwargs):
        bar = self.bars[phase.name]
        bar.set_postfix_str(fmt_losses(losses))
        bar.update(1)
        bar.refresh()

        for k, v in losses.items():
            self.counts[phase.name][k] += target_size
            self.values[phase.name][k] += target_size * v

    def epoch_ended(self, epoch, **kwargs):
        for bar in self.bars.values():
            bar.n = 0
            bar.refresh()

        for phase_key, phase_val in self.counts.items():
            phase_losses = {}
            for loss_key, loss_val in phase_val.items():
                metric = self.values[phase_key][loss_key] / self.counts[phase_key][loss_key]
                phase_losses[loss_key] = metric
            print(f'Epoch: {epoch:4d} | {phase_key} | {fmt_losses(phase_losses)}')
        self.init_counters()

    def training_ended(self, **kwargs):
        for bar in self.bars.values():
            bar.n = bar.total
            bar.refresh()
            bar.close()


def detach_state_dict(state_dict):
    """
    Detach state from the graph.
    """
    res = OrderedDict()
    for k, v in state_dict.items():
        res[k] = v.clone().detach()
    return res


def train(model, phases, opt, loss_fn, epochs=1, grad_clip=10, log_n_last_epochs=0):
    """
    A non-bptt training loop
    """
    pb = ProgressBar()
    pb.training_started(phases)

    state_dicts = []

    for epoch in range(1, epochs + 1):

        for phase in phases:
            is_training = phase.is_training
            model.train(is_training)

            for batch in phase.dataloader:

                endog, seas_mask, exog_cat, exog_cnt, target = batch

                with torch.set_grad_enabled(is_training):
                    out = model.forward(endog, seas_mask, exog_cat, exog_cnt, target)

                loss = out.loss_result.overall

                pb.batch_ended(phase, out.loss_result.losses, target_size=target.size(0))

                if is_training:
                    opt.zero_grad()
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                    opt.step()

            if is_training and log_n_last_epochs > 0:
                sd = detach_state_dict(model.state_dict())
                state_dicts.append(sd)

                if len(state_dicts) >= log_n_last_epochs:
                    del state_dicts[0]

        pb.epoch_ended(epoch)
    pb.training_ended()

    return state_dicts
