from collections import namedtuple
from itertools import chain

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter

import rpy2.robjects as ro

base = importr('base')
forecast = importr('forecast')


def get_r_model(ts, period1, period2, h, coeffs, armethod):

    with localconverter(ro.default_converter + numpy2ri.converter):
        r_ts = ro.conversion.py2rpy(ts)

    r_msts = forecast.msts(r_ts, **{
        'seasonal.periods': ro.IntVector([period1, period2]),
    })

    fit = forecast.dshw(r_msts,
                        armethod=armethod,
                        h=h, **coeffs)

    return convert_model_props(fit)


def convert_model_props(fit):
    fitd = dict(zip(fit.names, list(fit)))

    pnames = ['alpha', 'beta', 'gamma', 'omega', 'phi']
    snames = ['mean', 'fitted', 'residuals']

    with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
        model = namedtuple('Params', pnames + snames)(
            *chain(
                (float(fitd['model'].rx[pname][0]) for pname in pnames),
                (ro.conversion.rpy2py(fitd[sname]) for sname in snames),
            )
        )

    return model
