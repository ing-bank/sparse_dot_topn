import numpy as np
import pytest

_rng = np.random.Generator(np.random.PCG64DXSM(1483336117))


@pytest.fixture()
def rng():
    return _rng
