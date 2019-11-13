# sparse\_dot\_topn: 

**sparse\_dot\_topn** provides a fast way to performing a sparse matrix multiplication followed by top-n multiplication result selection.

Comparing very large feature vectors and picking the best matches, in practice often results in performing a sparse matrix multiplication followed by selecting the top-n multiplication results. In this package, we implement a customized Cython function for this purpose. When comparing our Cythonic approach to doing the same use with SciPy and NumPy functions, **our approach improves the speed by about 40% and reduces memory consumption.**

This package is made by ING Wholesale Banking Advanced Analytics team. This [blog](https://medium.com/@ingwbaa/https-medium-com-ingwbaa-boosting-selection-of-the-most-similar-entities-in-large-scale-datasets-450b3242e618) explains how we implement it.

## Example
``` python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import rand
from sparse_dot_topn import awesome_cossim_topn

N = 10
a = rand(100, 1000000, density=0.005, format='csr')
b = rand(1000000, 200, density=0.005, format='csr')

# Use standard implementation

c = awesome_cossim_topn(a, b, N, 0.01)

# Use parallel implementation with 4 threads

d = awesome_cossim_topn(a, b, N, 0.01, use_threads=True, n_jobs=4)
```

You can also find code which compares our boosting method with calling scipy+numpy function directly in example/comparison.py

## Dependency and Install
Install `numpy` and `cython` first before installing this package. Then,
``` sh
pip install sparse_dot_topn
```


## Uninstall
``` sh
pip uninstall sparse_dot_topn
```
