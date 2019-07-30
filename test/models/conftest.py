import pytest

from ..utils.fixtures import DataSetOneFixture


@pytest.fixture(scope="module")
def dataset_one():
    return DataSetOneFixture()
