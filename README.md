# sparse\_dot\_topn: 

**sparse\_dot\_topn** provides a fast way to performing a sparse matrix multiplication followed by top-n multiplication result selection.

Comparing very large feature vectors and picking the best matches, in practice often results in performing a sparse matrix multiplication followed by selecting the top-n multiplication results. In this package, we implement a customized Cython function for this purpose. When comparing our Cythonic approach to doing the same use with SciPy and NumPy functions, **our approach improves the speed by about 40% and reduces memory consumption.**

This package is made by ING Wholesale Banking Advanced Analytics team. This [blog](https://medium.com/@ingwbaa/https-medium-com-ingwbaa-boosting-selection-of-the-most-similar-entities-in-large-scale-datasets-450b3242e618) or this [blog](https://www.sun-analytics.nl/posts/2017-07-26-boosting-selection-of-most-similar-entities-in-large-scale-datasets/) explains how we implement it.

## Example
``` python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import rand
from sparse_dot_topn import awesome_cossim_topn

N = 10
a = rand(100, 1000000, density=0.005, format='csr')
b = rand(1000000, 200, density=0.005, format='csr')

# Default precision type is np.float64, but you can down cast to have a small memory footprint and faster execution
# Remark : These are the only 2 types supported now, since we assume that float16 will be difficult to implement and will be slower, because C doesn't support a 16-bit float type on most PCs
a = a.astype(np.float32)
b = b.astype(np.float32)

# Use standard implementation
c = awesome_cossim_topn(a, b, N, 0.01)

# Use parallel implementation with 4 threads
d = awesome_cossim_topn(a, b, N, 0.01, use_threads=True, n_jobs=4)

# Use standard implementation with 4 threads and with the computation of best_ntop: the value of ntop needed to capture all results above lower_bound
d, best_ntop = awesome_cossim_topn(a, b, N, 0.01, use_threads=True, n_jobs=4, return_best_ntop=True)
```

You can also find code which compares our boosting method with calling scipy+numpy function directly in example/comparison.py

## Dependency and Install
Install `numpy` and `cython` first before installing this package. Then,
``` sh
pip install sparse_dot_topn
```

From version >=0.3.0, we don't proactively support python 2.7. However, you should still be able to install this package in python 2.7.
If you encounter gcc compiling issue, please refer these discussions and setup CFLAGS and CXXFLAGS variables
- https://github.com/ing-bank/sparse_dot_topn/issues/7#issuecomment-695165663

## Uninstall
``` sh
pip uninstall sparse_dot_topn
```


## Local development

``` sh
python setup.py clean --all
python setup.py develop
pytest
```


``` sh
python -m build
cd dist/
pip install sparse_dot_topn-*.tar.gz
```
