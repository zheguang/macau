# Macau

Macau is one of the tools used in Zgraggen, Zhao, Zeleznik and Kraska, _Investigating the Effect of the Multiple Comparison Problem in Visual Analysis_ in CHI 2018.

Macau consists of two parts, the data generation and the statitical hypothesis testing.

## Data generation

To generate data, use `data_generator.py` script.  Use `-h` to see usage.

## Hypothesis testing

`macau.py` uses [resampling](testing://en.wikipedia.org/wiki/Resampling_(statistics)) to compute test statistics and _p_-values.

Usage:
```bash
python3 macau.py <exec_mode> <data_path> <permutations> <hypotheses_path>
```
The `data_path` is a data file output by `data_generator`.
The `hypotheses_path` is a file containing hypotheses using the encoding intoduced in [1].

## Reference
[1] Zgraggen, Zhao, Zeleznik and Kraska, _Investigating the Effect of the Multiple Comparison Problem in Visual Analysis_, CHI 2018.
