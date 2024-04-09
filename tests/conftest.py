import numpy as np
import pytest


@pytest.fixture(scope="package")
def rng():
    return np.random.Generator(np.random.PCG64DXSM(1481236117))
