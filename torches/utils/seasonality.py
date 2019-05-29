import torch


def pytorch_moving_average(a, n=2):
    assert a.ndimension() == 3, "3-dim input is expected"

    ret = torch.cumsum(a, dim=1)
    retc = ret.clone()
    retc[:, n:, :] = ret[:, n:, :] - ret[:, :-n, :]
    return retc[:, n - 1:, :] / n


# TODO: test this against R

# Calculating seasonal indexes
def mult_seasindex(y, p):

    assert y.ndimension() == 3, "3-dim input is expected"

    bs = y.size(0)
    n = y.size(1)
    f = y.size(2)

    n2 = 2 * p
    shorty = y[:, :n2, :]
    average = torch.empty((bs, n, f))
    simplema = pytorch_moving_average(shorty, p)

    if p % 2 == 0:  # Even order
        centeredma = pytorch_moving_average(simplema[:, 0:(n2 - p + 1), :], 2)
#         print(shorty.shape, average[p//2:p].shape, shorty[p//2:p].shape, centeredma[:p].shape)
        offset = p // 2
        averagec = average.clone()
        averagec[:, offset:offset + p, :] = shorty[:, offset:offset + p, :] / centeredma[:, :p, :]
        average = averagec
        ii = list(range(p, (p + p // 2))) + list(range((p // 2), p))
        si = average[:, ii, :]  # average[[p + (:(p / 2)), (p / 2):p]]
    else:
        offset = (p - 1) // 2
        average[:, offset:offset + p, :] = shorty[:, offset:offset + p, :] / simplema[:, :p, :]
        ii = list(range(p, (p + (p - 1) // 2))) + list(range((p // 2), p))
        si = average[:, ii, :]
        raise Exception("Not tested")
    return si


def add_seasindex(y, p):

    assert y.ndimension() == 3, "3-dim input is expected"

    bs = y.size(0)
    n = y.size(1)
    f = y.size(2)

    n2 = 2 * p
    shorty = y[:, :n2, :]
    average = torch.empty((bs, n, f))
    simplema = pytorch_moving_average(shorty, p)

    if p % 2 == 0:  # Even order
        centeredma = pytorch_moving_average(simplema[:, 0:(n2 - p + 1), :], 2)
#         print(shorty.shape, average[p//2:p].shape, shorty[p//2:p].shape, centeredma[:p].shape)
        offset = p // 2
        averagec = average.clone()
        averagec[:, offset:offset + p, :] = shorty[:, offset:offset + p, :] - centeredma[:, :p, :]
        average = averagec
        ii = list(range(p, (p + p // 2))) + list(range((p // 2), p))
        si = average[:, ii, :]  # average[[p + (:(p / 2)), (p / 2):p]]
    else:
        offset = (p - 1) // 2
        average[:, offset:offset + p, :] = shorty[:, offset:offset + p, :] - simplema[:, :p, :]
        ii = list(range(p, (p + (p - 1) // 2))) + list(range((p // 2), p))
        si = average[:, ii, :]
        raise Exception("Not tested")
    return si
