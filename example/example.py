from scipy.sparse import rand
from sparse_dot_topn import awesome_cossim_topn

N = 10
a = rand(100, 1000000, density=0.005, format='csr')
b = rand(1000000, 200, density=0.005, format='csr')

# Use standard implementation

c = awesome_cossim_topn(a, b, 5, 0.01)

# Use parallel implementation with 4 threads

d = awesome_cossim_topn(a, b, 5, 0.01, use_threads=True, n_jobs=4)
