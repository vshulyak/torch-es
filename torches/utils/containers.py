from torch.utils.data import DataLoader

from dataclasses import dataclass


@dataclass
class Phase():
    name: str
    is_training: bool
    dataloader: DataLoader


@dataclass
class LossResult():
    overall: float
    losses: dict


def make_obj(**kwargs):
    return type('Res', (object,), kwargs)()
