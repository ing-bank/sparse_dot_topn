# Benchmark sparse-dot-topn

Minor benchmark sweet to compare against Scipy's sparse matrix multiplication

## Dependencies

```shell
pip install richbench
```

## Data

List of company names from the EDGAR company database is used as a reference set.
You can download it at: `https://www.kaggle.com/datasets/dattapiy/sec-edgar-companies-list?select=sec__edgar_company_info.csv`.
The script expects the csv with the name: `sec__edgar_company_info.csv` or you can set the filename on line 19 in `bench_scipy_csr.py`.
A and B are TF-IDF matrices over trigrams of the company names.

You can change the number of rows in the matrices by setting `N_ROWS` in `bench_scipy_csr.py` on line 18.

### Reference

```shell
richbench /bench --repeat 10 --times 10
```

Benchmark Scipy 1.12.0 vs sparse-dot-topn v1.0.0 
Apple M2 Pro

Time is 10 times `A.dot(B)` with shapes `(20_000, 193190) x (193190, 20_000)` repeated 10 times.

| Benchmark               | top_n | n_threads | Min     | Max     | Mean  | Min (+)        | Max (+)        | Mean (+)       |
| :-----------------------| :---: | :-------: | :------ | :------ | :---- | :------------- | :------------- | :------------- |
| Scipy vs sp_matmul      |       | 1         | 1.631   | 1.665   | 1.644 | 1.637 (1.0x)   | 1.671 (1.0x)   | 1.645 (1.0x)   |
| Scipy vs sp_matmul      |       | 1         | 1.631   | 1.665   | 1.644 | 1.637 (1.0x)   | 1.671 (1.0x)   | 1.645 (1.0x)   |
| Scipy vs sp_matmul      |       | 2         | 1.636   | 1.648   | 1.640 | 0.850 (1.9x)   | 0.863 (1.9x)   | 0.856 (1.9x)   |
| Scipy vs sp_matmul      |       | 4         | 1.633   | 1.653   | 1.643 | 0.494 (3.3x)   | 0.499 (3.3x)   | 0.497 (3.3x)   |
| Scipy vs sp_matmul      |       | 8         | 1.631   | 1.664   | 1.644 | 0.320 (5.1x)   | 0.322 (5.2x)   | 0.321 (5.1x)   |
| Scipy vs sp_matmul_topn | 10    | 1         | 1.628   | 1.647   | 1.637 | 1.634 (1.0x)   | 1.653 (1.0x)   | 1.641 (1.0x)   |
| Scipy vs sp_matmul_topn | 20    | 1         | 1.637   | 1.671   | 1.644 | 1.855 (-1.1x)  | 1.895 (-1.1x)  | 1.872 (-1.1x)  |
| Scipy vs sp_matmul_topn | 30    | 1         | 1.635   | 1.672   | 1.643 | 1.987 (-1.2x)  | 2.003 (-1.2x)  | 1.996 (-1.2x)  |
| Scipy vs sp_matmul_topn | 100   | 1         | 1.631   | 1.680   | 1.641 | 3.244 (-2.0x)  | 3.256 (-1.9x)  | 3.250 (-2.0x)  |
| Scipy vs sp_matmul_topn | 1000  | 1         | 1.634   | 1.656   | 1.642 | 10.262 (-6.3x) | 10.446 (-6.3x) | 10.354 (-6.3x) |
| Scipy vs sp_matmul_topn | 10    | 2         | 1.633   | 1.648   | 1.639 | 0.778 (2.1x)   | 0.790 (2.1x)   | 0.785 (2.1x)   |
| Scipy vs sp_matmul_topn | 20    | 2         | 1.636   | 1.664   | 1.643 | 0.899 (1.8x)   | 0.913 (1.8x)   | 0.905 (1.8x)   |
| Scipy vs sp_matmul_topn | 30    | 2         | 1.635   | 1.651   | 1.641 | 0.967 (1.7x)   | 0.977 (1.7x)   | 0.971 (1.7x)   |
| Scipy vs sp_matmul_topn | 100   | 2         | 1.639   | 1.672   | 1.645 | 1.658 (-1.0x)  | 1.675 (-1.0x)  | 1.663 (-1.0x)  |
| Scipy vs sp_matmul_topn | 1000  | 2         | 1.627   | 1.648   | 1.638 | 5.719 (-3.5x)  | 5.747 (-3.5x)  | 5.731 (-3.5x)  |
| Scipy vs sp_matmul_topn | 10    | 4         | 1.638   | 1.650   | 1.641 | 0.420 (3.9x)   | 0.424 (3.9x)   | 0.422 (3.9x)   |
| Scipy vs sp_matmul_topn | 20    | 4         | 1.639   | 1.658   | 1.645 | 0.485 (3.4x)   | 0.504 (3.3x)   | 0.490 (3.4x)   |
| Scipy vs sp_matmul_topn | 30    | 4         | 1.632   | 1.691   | 1.644 | 0.522 (3.1x)   | 0.525 (3.2x)   | 0.524 (3.1x)   |
| Scipy vs sp_matmul_topn | 100   | 4         | 1.638   | 1.651   | 1.642 | 0.892 (1.8x)   | 0.899 (1.8x)   | 0.895 (1.8x)   |
| Scipy vs sp_matmul_topn | 1000  | 4         | 1.636   | 1.651   | 1.644 | 3.164 (-1.9x)  | 3.177 (-1.9x)  | 3.171 (-1.9x)  |
| Scipy vs sp_matmul_topn | 10    | 8         | 1.637   | 1.666   | 1.645 | 0.244 (6.7x)   | 0.248 (6.7x)   | 0.246 (6.7x)   |
| Scipy vs sp_matmul_topn | 20    | 8         | 1.636   | 1.655   | 1.642 | 0.281 (5.8x)   | 0.286 (5.8x)   | 0.282 (5.8x)   |
| Scipy vs sp_matmul_topn | 30    | 8         | 1.631   | 1.650   | 1.639 | 0.302 (5.4x)   | 0.304 (5.4x)   | 0.303 (5.4x)   |
| Scipy vs sp_matmul_topn | 100   | 8         | 1.633   | 1.664   | 1.643 | 0.511 (3.2x)   | 0.514 (3.2x)   | 0.513 (3.2x)   |
| Scipy vs sp_matmul_topn | 1000  | 8         | 1.632   | 1.650   | 1.642 | 1.904 (-1.2x)  | 1.910 (-1.2x)  | 1.906 (-1.2x)  |
