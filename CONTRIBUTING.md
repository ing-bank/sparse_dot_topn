# Contributing

Contributions are welcome!
Please open an [issue](https://github.com/ing-bank/sparse_dot_topn/issues)
or create a [pull request](https://github.com/ing-bank/sparse_dot_topn/pulls).

## Installation

To install the testing dependencies:

```shell
pip install -e .[test]
```

Note that you're building from source and need to have a C++17 compatible compiler available.
If you will be working on C++ code you should install with:

```shell
SKBUILD_CMAKE_ARGS="-DSDTN_ENABLE_DEVMODE=ON" pip install sparse_dot_topn --no-binary sparse_dot_topn
```

## Commit message convention

`sparse_dot_topn` follows a [numpy-inspired](https://numpy.org/devdocs/dev/development_workflow.html#writing-the-commit-message) commit message convention:

```text
API: an (incompatible) API change
BENCH: changes to the benchmark suite
BLD/PKG: change related to building sparse_dot_topn
BUG: bug fix
CHG: functional code change
CICD: CI/CD changes
DEP: deprecate something, or remove a deprecated object
DEV: development tool or utility
DOC: documentation
ENH: enhancement
MAINT: maintenance commit (refactoring, typos, etc.)
REV: revert an earlier commit
STY: style fix (whitespace, PEP8)
TST: addition or modification of tests
TYP: static typing
REL: related to releasing
```

If your commit touches C++ code please add `[C++]` after the prefix like `CHG: [C++] <commit message>`

## Testing

To run the tests:

```shell
cd tests
pytest
```
