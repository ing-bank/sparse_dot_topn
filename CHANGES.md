# Release history:


## v0.3.1
- Adding the possibility to use smaller data type float32 instead of the default float64
- Adding unit tests to cover both data types
- Adding unit tests to cover return_best_ntop==False

## v0.3.0
- defragmented memory used during computation [PR-53](https://github.com/ing-bank/sparse_dot_topn/pull/53)
- Enable Github Action for unit test
- add license metadata [PR-47](https://github.com/ing-bank/sparse_dot_topn/pull/47)

## v0.2.9
- added unit tests. 
- awesome_cossim_topn checks for zero input matrices.
- fix flake8 warnings.

## v0.2.8
- quick fix for nt installation problem [ISSUE-26]

## v0.2.7
- add n_jobs value validation
- matrix dimension validation
- change windows mingw32 compiler
- fix numpy and Cython install dependency

## v0.2.6
- import parallel implementation

## v0.2.5
- fix python 2 install problem

## v0.2.5
- fix python 2 install problem, but failed

## v0.2.3
- Make long description use markdown format

## v0.2.2
- update readme file name

## v0.2.1
- fix the problem when pip install by tar.gz;
- update meta-information of package

## v0.2, Nov 14, 2018:
- improve the function import layer. Now awesome_cossim_topn can be used directly after `from sparse_dot_topn import awesome_cossim_topn`.
- improve the module installation

## v0.1, July 13, 2017:
- Release the first version

