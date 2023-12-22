# Copyright (c) 2023 ING Analytics Wholesale Banking
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.types import DTypeLike, NDArray
    from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

__all__ = ["assert_idx_dtype", "assert_supported_dtype", "ensure_compatible_dtype", "is_supported_dtype"]

_SUPPORTED_DTYPES = {np.dtype("int32"), np.dtype("int64"), np.dtype("float32"), np.dtype("float64")}


def assert_idx_dtype(dtype: DTypeLike | None) -> DTypeLike:
    if dtype is None:
        return np.dtype("int32")
    if not (dtype == np.dtype("int32") or dtype == np.dtype("int64")):
        msg = f"`idx_dtype` must be a 32 or 64 bit integer, got {dtype}."
        raise TypeError(msg)
    return dtype


def assert_supported_dtype(obj: NDArray | coo_matrix | csc_matrix | csr_matrix, name: str | None = None):
    if obj.dtype in _SUPPORTED_DTYPES:
        return
    msg = "Supported dtypes are {32, 64}bit {int, float}" + f" got {name or 'obj'}.dtype: {obj.dtype}"
    raise TypeError(msg)


def ensure_compatible_dtype(
    a: NDArray | coo_matrix | csc_matrix | csr_matrix, b: NDArray | coo_matrix | csc_matrix | csr_matrix
):
    """Check and convert dtype if needed and safe to do.

    When the dtypes match nothing is done.
    If one has a lower precision dtype of the same kind as the other, the lower precision object
    is cast to the higher precision one. Otherwhise an error is raised.

    """
    if a.dtype == b.dtype:
        return a, b
    if a.dtype.itemsize > b.dtype.itemsize:
        lhs = b
        rhs = a
    else:
        lhs = a
        rhs = b
    if np.issubdtype(lhs.dtype, rhs.dtype):
        return lhs.astype(rhs.dtype), rhs
    msg = "`a` and `b` do not have the same dtype and cannot be safely cast"
    raise TypeError(msg)


def is_supported_dtype(dtype: DTypeLike) -> bool:
    return dtype in _SUPPORTED_DTYPES
