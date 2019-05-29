import numpy as np

import torch

import pytest

from torches.modules.dshw_module import DSHWModule
from .utils.r_model import get_r_model


def test_r_vs_torch_noar(dataset_one, model_coeffs_noar):

    armethod = False

    r_reference_model = get_r_model(dataset_one.ts,
                                    dataset_one.period1,
                                    dataset_one.period2,
                                    dataset_one.h,
                                    model_coeffs_noar,
                                    armethod=armethod)

    torch_model = DSHWModule(input_dim=1,
                             period1=dataset_one.period1,
                             period2=dataset_one.period2,
                             hs=[dataset_one.h],
                             enable_ar=armethod,
                             alphas=[model_coeffs_noar['alpha']],
                             betas=[model_coeffs_noar['beta']],
                             gammas=[model_coeffs_noar['gamma']],
                             omegas=[model_coeffs_noar['omega']],
                             phis=[model_coeffs_noar['phi']])(dataset_one.torch_ts.view(1, -1, 1))

    assert np.allclose(torch_model.mean.detach().numpy().squeeze(), r_reference_model.mean)
    assert np.allclose(torch_model.yhat.detach().numpy().squeeze(), r_reference_model.fitted)


def test_r_vs_torch_ar(dataset_one, model_coeffs_ar):

    armethod = True

    r_reference_model = get_r_model(dataset_one.ts,
                                    dataset_one.period1,
                                    dataset_one.period2,
                                    dataset_one.h,
                                    model_coeffs_ar,
                                    armethod=armethod)

    torch_model = DSHWModule(input_dim=1,
                             period1=dataset_one.period1,
                             period2=dataset_one.period2,
                             hs=[dataset_one.h],
                             enable_ar=armethod,
                             alphas=[model_coeffs_ar['alpha']],
                             betas=[model_coeffs_ar['beta']],
                             gammas=[model_coeffs_ar['gamma']],
                             omegas=[model_coeffs_ar['omega']],
                             phis=[model_coeffs_ar['phi']])(dataset_one.torch_ts.view(1, -1, 1))

    assert np.allclose(torch_model.mean.detach().numpy().squeeze(), r_reference_model.mean)
    assert np.allclose(torch_model.yhat.detach().numpy().squeeze(), r_reference_model.fitted)


@pytest.mark.parametrize("is_multiplicative", [True, False])
def test_gradient_flow(is_multiplicative, dataset_one, model_coeffs_ar):

    torch_model = DSHWModule(input_dim=1,
                             period1=dataset_one.period1,
                             period2=dataset_one.period2,
                             hs=[dataset_one.h],
                             enable_ar=True,
                             is_multiplicative=is_multiplicative,
                             alphas=[model_coeffs_ar['alpha']],
                             betas=[model_coeffs_ar['beta']],
                             gammas=[model_coeffs_ar['gamma']],
                             omegas=[model_coeffs_ar['omega']],
                             phis=[model_coeffs_ar['phi']])(dataset_one.torch_ts.view(1, -1, 1))

    mse = torch.mean(torch_model.e[:, :, 0]**2)
    mse.backward()  # in case of in-place modification, this will raise RuntimeError


@pytest.mark.parametrize("enable_ar", [True, False])
@pytest.mark.parametrize("is_multiplicative", [True, False])
def test_two_series(enable_ar, is_multiplicative, dataset_2_feature, model_coeffs_ar):

    ref_1 = DSHWModule(input_dim=1,
                       period1=dataset_2_feature.period1,
                       period2=dataset_2_feature.period2,
                       hs=[dataset_2_feature.h],
                       enable_ar=enable_ar,
                       is_multiplicative=is_multiplicative,
                       alphas=[model_coeffs_ar['alpha']],
                       betas=[model_coeffs_ar['beta']],
                       gammas=[model_coeffs_ar['gamma']],
                       omegas=[model_coeffs_ar['omega']],
                       phis=[model_coeffs_ar['phi']])(dataset_2_feature.torch_ts[:, 0].view(1, -1, 1))

    ref_2 = DSHWModule(input_dim=1,
                       period1=dataset_2_feature.period1,
                       period2=dataset_2_feature.period2,
                       hs=[dataset_2_feature.h],
                       enable_ar=enable_ar,
                       is_multiplicative=is_multiplicative,
                       alphas=[model_coeffs_ar['alpha']],
                       betas=[model_coeffs_ar['beta']],
                       gammas=[model_coeffs_ar['gamma']],
                       omegas=[model_coeffs_ar['omega']],
                       phis=[model_coeffs_ar['phi']])(dataset_2_feature.torch_ts[:, 1].view(1, -1, 1))

    ref_all = DSHWModule(input_dim=2,
                         period1=dataset_2_feature.period1,
                         period2=dataset_2_feature.period2,
                         hs=[dataset_2_feature.h, dataset_2_feature.h],
                         enable_ar=enable_ar,
                         is_multiplicative=is_multiplicative,
                         alphas=[model_coeffs_ar['alpha'], model_coeffs_ar['alpha']],
                         betas=[model_coeffs_ar['beta'], model_coeffs_ar['beta']],
                         gammas=[model_coeffs_ar['gamma'], model_coeffs_ar['gamma']],
                         omegas=[model_coeffs_ar['omega'], model_coeffs_ar['omega']],
                         phis=[model_coeffs_ar['phi'], model_coeffs_ar['phi']]
                         )(dataset_2_feature.torch_ts.view(1, -1, 2))

    assert torch.allclose(torch.cat([ref_1.mean, ref_2.mean], dim=2), ref_all.mean)
    assert torch.allclose(torch.cat([ref_1.yhat, ref_2.yhat], dim=2), ref_all.yhat)
    assert torch.allclose(torch.cat([ref_1.e, ref_2.e], dim=2), ref_all.e)


@pytest.mark.parametrize("enable_ar", [True, False])
@pytest.mark.parametrize("is_multiplicative", [True, False])
def test_2_batches(enable_ar, is_multiplicative, dataset_2_batch, model_coeffs_ar):

    ref_1 = DSHWModule(input_dim=1,
                       period1=dataset_2_batch.period1,
                       period2=dataset_2_batch.period2,
                       hs=[dataset_2_batch.h],
                       enable_ar=enable_ar,
                       is_multiplicative=is_multiplicative,
                       alphas=[model_coeffs_ar['alpha']],
                       betas=[model_coeffs_ar['beta']],
                       gammas=[model_coeffs_ar['gamma']],
                       omegas=[model_coeffs_ar['omega']],
                       phis=[model_coeffs_ar['phi']])(dataset_2_batch.torch_ts[0, :].view(1, -1, 1))

    ref_2 = DSHWModule(input_dim=1,
                       period1=dataset_2_batch.period1,
                       period2=dataset_2_batch.period2,
                       hs=[dataset_2_batch.h],
                       enable_ar=enable_ar,
                       is_multiplicative=is_multiplicative,
                       alphas=[model_coeffs_ar['alpha']],
                       betas=[model_coeffs_ar['beta']],
                       gammas=[model_coeffs_ar['gamma']],
                       omegas=[model_coeffs_ar['omega']],
                       phis=[model_coeffs_ar['phi']])(dataset_2_batch.torch_ts[1, :].view(1, -1, 1))

    ref_all = DSHWModule(input_dim=1,
                         period1=dataset_2_batch.period1,
                         period2=dataset_2_batch.period2,
                         hs=[dataset_2_batch.h],
                         enable_ar=enable_ar,
                         is_multiplicative=is_multiplicative,
                         alphas=[model_coeffs_ar['alpha']],
                         betas=[model_coeffs_ar['beta']],
                         gammas=[model_coeffs_ar['gamma']],
                         omegas=[model_coeffs_ar['omega']],
                         phis=[model_coeffs_ar['phi']]
                         )(dataset_2_batch.torch_ts.view(2, -1, 1))

    assert torch.allclose(torch.cat([ref_1.mean, ref_2.mean], dim=0), ref_all.mean)
    assert torch.allclose(torch.cat([ref_1.yhat, ref_2.yhat], dim=0), ref_all.yhat)
    assert torch.allclose(torch.cat([ref_1.e, ref_2.e], dim=0), ref_all.e)


@pytest.mark.parametrize("enable_ar", [True, False])
@pytest.mark.parametrize("is_multiplicative", [True, False])
def test_two_batches_two_series(enable_ar, is_multiplicative, dataset_2_batch_2_feature, model_coeffs_ar):

    ref_1 = DSHWModule(input_dim=1,
                       period1=dataset_2_batch_2_feature.period1,
                       period2=dataset_2_batch_2_feature.period2,
                       hs=[dataset_2_batch_2_feature.h],
                       enable_ar=enable_ar,
                       is_multiplicative=is_multiplicative,
                       alphas=[model_coeffs_ar['alpha']],
                       betas=[model_coeffs_ar['beta']],
                       gammas=[model_coeffs_ar['gamma']],
                       omegas=[model_coeffs_ar['omega']],
                       phis=[model_coeffs_ar['phi']])(dataset_2_batch_2_feature.torch_ts[:, :, 0:1])

    ref_2 = DSHWModule(input_dim=1,
                       period1=dataset_2_batch_2_feature.period1,
                       period2=dataset_2_batch_2_feature.period2,
                       hs=[dataset_2_batch_2_feature.h],
                       enable_ar=enable_ar,
                       is_multiplicative=is_multiplicative,
                       alphas=[model_coeffs_ar['alpha']],
                       betas=[model_coeffs_ar['beta']],
                       gammas=[model_coeffs_ar['gamma']],
                       omegas=[model_coeffs_ar['omega']],
                       phis=[model_coeffs_ar['phi']])(dataset_2_batch_2_feature.torch_ts[:, :, 1:2])

    ref_all = DSHWModule(input_dim=2,
                         period1=dataset_2_batch_2_feature.period1,
                         period2=dataset_2_batch_2_feature.period2,
                         hs=[dataset_2_batch_2_feature.h, dataset_2_batch_2_feature.h],
                         enable_ar=enable_ar,
                         is_multiplicative=is_multiplicative,
                         alphas=[model_coeffs_ar['alpha'], model_coeffs_ar['alpha']],
                         betas=[model_coeffs_ar['beta'], model_coeffs_ar['beta']],
                         gammas=[model_coeffs_ar['gamma'], model_coeffs_ar['gamma']],
                         omegas=[model_coeffs_ar['omega'], model_coeffs_ar['omega']],
                         phis=[model_coeffs_ar['phi'], model_coeffs_ar['phi']]
                         )(dataset_2_batch_2_feature.torch_ts)

    assert torch.allclose(torch.cat([ref_1.mean, ref_2.mean], dim=2), ref_all.mean)
    assert torch.allclose(torch.cat([ref_1.yhat, ref_2.yhat], dim=2), ref_all.yhat)
    assert torch.allclose(torch.cat([ref_1.e, ref_2.e], dim=2), ref_all.e)
