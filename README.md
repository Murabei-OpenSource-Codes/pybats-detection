## Welcome to `pybats-detection`

The `pybats-detection` is a `python` package with routines implemented in python for detection of outlier and structural changes in time series using Bayesian Dynamic Linear Models (DLM).
The currently version of the package implements the automatic monitoring, manual intervention and smoothing for DLM’s.

The stable version of `pybats-detection` can be installed from [PyPI](https://pypi.org/) using:

```
pip install pybats-detection
```

The development version can be installed from [GitHub](https://github.com/) using:

```
git clone git@github.com:Murabei-OpenSource-Codes/develop/pybats-detection.git pybats-detection
cd pybats-detection
python setup.py install
```

The package uses the `pybats.dglm.dlm` objects from [`PyBATS`](https://github.com/lavinei/pybats) package as an input for the following classes:

- `Monitoring`: perform automatic monitoring of outlier and/or structural changes in time series according to [West and Harisson (1986)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1986.10478331) .

- `Intervention`: perform manual intervention of outlier and/or structural changes in time series according to [West and Harrison (1989)](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.3980080104).

- `Smoothing`: compute the retrospective state space parameter and predictive distributions.

All three classes have the `fit` method which received the univariate time series
as a `pandas.Series` object and further arguments related to each class.

User manuals can be found in:

- [pybats_detection](https://raw.githubusercontent.com/Murabei-OpenSource-Codes/pybats-detection/develop/vignettes/pybats_detection.pdf): detailed explanation of `pybats-detection` usability.

- [quick_start](https://raw.githubusercontent.com/Murabei-OpenSource-Codes/pybats-detection/develop/vignettes/quick_start.pdf): quick reference guide with step-by-step usability.

## Authors

`pybats-detection` was developed by [André Menezes](https://andrmenezes.github.io/) and
[Eduardo Gabriel](https://www.linkedin.com/in/eduardo-gabriel-433332142/)
while working as Data Scientist at [Murabei Data Science](https://www.murabei.com/)
advised by professor [Hélio Migon](http://lattes.cnpq.br/7997248190492823) and
[André Baceti](https://br.linkedin.com/in/andre-baceti/pt) .


## License

The `pybats-detection` package is released under the Apache License, Version 2.0.
Please, see file [`LICENSE.md`](https://github.com/Murabei-OpenSource-Codes/pybats-detection/blob/develop/LICENSE.md).
