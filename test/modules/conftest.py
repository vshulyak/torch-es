import pytest

import rpy2.robjects as ro

from ..utils.fixtures import DataSetOneFixture, DataSet2FeatureFixture, DataSet2BatchFixture, DataSet2Batch2FeatureFixture


@pytest.fixture(scope="module")
def dataset_one():
    return DataSetOneFixture()


@pytest.fixture(scope="module")
def dataset_2_feature():
    return DataSet2FeatureFixture()


@pytest.fixture(scope="module")
def dataset_2_batch():
    return DataSet2BatchFixture()


@pytest.fixture(scope="module")
def dataset_2_batch_2_feature():
    return DataSet2Batch2FeatureFixture()


@pytest.fixture(scope="module")
def model_coeffs_noar():
    return {
        'alpha': 0.1648906365819534,
        'beta': 0.08265189925077121,
        'gamma': 2.398595300137416e-07,
        'omega': 0.2309265700845013,
        'phi': 0.0,
        'lambda': ro.r("NULL")
    }


@pytest.fixture(scope="module")
def model_coeffs_ar():
    return {
        'alpha': 0.1648906365819534,
        'beta': 0.08265189925077121,
        'gamma': 2.398595300137416e-07,
        'omega': 0.2309265700845013,
        'phi': 0.1448967,
        'lambda': ro.r("NULL")
    }
