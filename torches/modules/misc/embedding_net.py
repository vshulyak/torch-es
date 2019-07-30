import torch
from torch import nn


class EmbeddingNet(nn.Module):
    """
    Applies embedding lookup for every feature
    """
    def __init__(self, conf, max_norm=None, norm_type=2.0, scale_grad_by_freq=False):
        super().__init__()
        embs = []
        for e_values, e_size in conf:
            e = nn.Embedding(e_values, e_size,
                             max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq)
            embs.append(e)
        self.embs = nn.ModuleList(embs)

    def forward(self, inputs):
        assert inputs.ndimension() == 2
        assert len(self.embs) == inputs.size(1)

        if not len(self.embs):
            return None

        inputs = inputs.type(torch.long)

        outs = []
        for i, emb in enumerate(self.embs):
            x = inputs[:, i].unsqueeze(1)
            x = emb(x).squeeze()
            outs.append(x)
        return torch.cat(outs, dim=1)
