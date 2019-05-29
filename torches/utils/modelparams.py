import torch


def pytorch_diff(inp):
    assert inp.ndimension() == 3, "3-dim input is expected"
    return inp[:, 1:, :] - inp[:, :-1, :]


def default_param(input_dim, param_value, default_value, requires_grad=True):
    if param_value is None:
        return siginv(torch.tensor([default_value] * input_dim, requires_grad=requires_grad))
    else:
        assert len(param_value) == input_dim
        return siginv(torch.tensor([param_value], requires_grad=requires_grad))


def siginv(x):
    return torch.log(x / (1 - x))
