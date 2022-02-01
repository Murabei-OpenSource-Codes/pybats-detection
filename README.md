
<!-- README.md is generated from README.Rmd. Please edit that file -->

## `pybats-detection`

> The `pybats-detection` is a `python` package with routines for
> detection of outlier and structural changes in time series using
> Bayesian Dynamic Linear Models (DLM). The currently version of the
> package implements the automatic monitoring, manual intervention and
> smoothing for DLM’s.

## Installation

The development version can be installed from
[GitHub](https://github.com/) using:

``` bash
$ git clone git@github.com:Murabei-OpenSource-Codes/pybats-detection.git pybats-detection
$ cd pybats-detection
$ python setup.py install
```

## Quick overview

The package uses the `pybats.dglm.dlm` objects from
[`PyBATS`](https://github.com/lavinei/pybats) package as an input for
the following classes:

-   `Monitoring`: perform automatic monitoring of outlier and/or
    structural changes in time series according to [West and
    Harisson (1986)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1986.10478331)
    .

-   `Intervention`: perform manual intervention of outlier and/or
    structural changes in time series according to [West and
    Harrison (1989)](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.3980080104).

-   `Smoothing`: compute the retrospective state space and predictive
    distributions.

All three classes have the `fit` method which received the univariate
time series as a `pandas.Series` object and further arguments related to
each class.

### Example of Monitoring

``` python
>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> from pybats.dglm import dlm
>>> from matplotlib.pyplot import figure
>>> from pybats_detection.loader import load_cp6
>>> from pybats_detection.monitor import Monitoring
>>> from pybats_detection.loader import load_telephone_calls
>>> from pybats_detection.intervention import Intervention
>>> 
>>> # Load data
>>> telephone_calls = load_telephone_calls()
>>> 
>>> # Defining the model
>>> a = np.array([350, 0])
>>> R = np.eye(2)
>>> np.fill_diagonal(R, val=[100])
>>> mod = dlm(a, R, ntrend=2, deltrend=0.95)
>>> 
>>> # Fitting with the automatic monitoring
>>> monitor = Monitoring(mod=mod, bilateral=True, prior_length=20)
>>> fit_monitor = monitor.fit(y=telephone_calls["average_daily_calls"], h=4,
>>>                           tau=0.135, change_var=[10, 2])
```

    ## Upper potential outlier detected at time 24 with H=6.1828e-03, L=6.1828e-03 and l=1
    ## Upper potential outlier detected at time 36 with H=4.6950e-02, L=4.6950e-02 and l=1
    ## Upper potential outlier detected at time 48 with H=1.0667e-02, L=1.0667e-02 and l=1
    ## Upper parametric change detected at time 61 with H=3.7151e+02, L=8.6657e-01 and l=3
    ## Lower parametric change detected at time 69 with H=7.3490e+01, L=2.1113e+01 and l=3
    ## Upper parametric change detected at time 73 with H=1.0179e+03, L=5.1031e+00 and l=3
    ## Lower potential outlier detected at time 77 with H=4.7737e-04, L=4.7737e-04 and l=1
    ## Lower potential outlier detected at time 79 with H=8.2908e-05, L=8.2908e-05 and l=1
    ## Upper potential outlier detected at time 84 with H=5.8788e-02, L=5.8788e-02 and l=1
    ## Upper potential outlier detected at time 95 with H=6.8930e-02, L=6.8930e-02 and l=1
    ## Upper potential outlier detected at time 108 with H=7.2666e-02, L=7.2666e-02 and l=1
    ## Lower potential outlier detected at time 115 with H=1.2539e-04, L=1.2539e-04 and l=1
    ## Lower parametric change detected at time 121 with H=9.8175e+00, L=9.8175e+00 and l=3
    ## Upper potential outlier detected at time 132 with H=2.8683e-02, L=2.8683e-02 and l=1
    ## Upper parametric change detected at time 137 with H=1.3781e+00, L=1.4891e-02 and l=3
    ## Lower potential outlier detected at time 138 with H=9.4740e-03, L=9.4740e-03 and l=1
    ## Lower potential outlier detected at time 140 with H=2.8301e-02, L=2.8301e-02 and l=1
    ## Lower potential outlier detected at time 141 with H=1.5974e-03, L=1.5974e-03 and l=1
    ## Upper potential outlier detected at time 144 with H=1.1437e-01, L=1.1437e-01 and l=1
    ## Lower potential outlier detected at time 146 with H=9.9659e-06, L=9.9659e-06 and l=1
    ## Lower potential outlier detected at time 147 with H=1.3965e-08, L=1.3965e-08 and l=1

``` python
>>> dict_filter = fit_monitor.get("filter")
>>> dict_filter.keys()
```

    ## dict_keys(['predictive', 'posterior'])

``` python
>>> data_predictive = dict_filter.get("predictive")
>>> data_predictive.head()
```

    ##    t  prior    y           f  ...  l_upper  what_detected    ci_lower    ci_upper
    ## 0  1   True  350  350.000000  ...        1        nothing  222.304223  477.695777
    ## 1  2   True  339  350.000000  ...        1        nothing  318.483933  381.516067
    ## 2  3   True  351  328.311860  ...        1        nothing  321.629479  334.994242
    ## 3  4   True  364  348.024290  ...        1        nothing  324.103892  371.944688
    ## 4  5   True  369  365.042878  ...        1        nothing  341.620552  388.465204
    ## 
    ## [5 rows x 16 columns]

``` python
>>> figure(figsize=(12, 8))
>>> plt.plot(telephone_calls["time"], telephone_calls["average_daily_calls"], "o",
>>>          markersize=4, color="black", fillstyle="none")
>>> # plt.plot(df_fit["f"], color="red")
>>> plt.plot(telephone_calls["time"], data_predictive["f"], color="blue")
>>> plt.plot(telephone_calls["time"], data_predictive["ci_lower"], color="blue",
>>>          linestyle="dashed")
>>> plt.plot(telephone_calls["time"], data_predictive["ci_upper"], color="blue",
>>>          linestyle="dashed")
>>> plt.grid(linestyle="dotted")
>>> plt.xlabel("Time")
>>> plt.ylabel("Average daily calls")
>>> plt.show()
```

<img src="examples/figures/README-plot-monitor-1.svg" style="display: block; margin: auto;" />

### Example of Intervention

``` python
>>> # Load data
>>> cp6 = load_cp6()
>>> 
>>> # Defining the model
>>> a = np.array([600, 1])
>>> R = np.array([[100, 0], [0, 25]])
>>> mod = dlm(a, R, ntrend=2, deltrend=[0.90, 0.98])
>>> 
>>> # Specifying the interventions
>>> list_interventions = [
>>>     {"time_index": 12, "which": ["variance", "noise"],
>>>      "parameters": [{"v_shift": "ignore"},
>>>                     {"h_shift": np.array([0, 0]),
>>>                      "H_shift": np.array([[1000, 25], [25, 25]])}]
>>>      },
>>>     {"time_index": 25, "which": ["noise", "variance"],
>>>      "parameters": [{"h_shift": np.array([80, 0]),
>>>                      "H_shift": np.array([[100, 0], [0, 0]])},
>>>                     {"v_shift": "ignore"}]},
>>>     {"time_index": 37, "which": ["subjective"],
>>>      "parameters": [{"a_star": np.array([970, 0]),
>>>                      "R_star": np.array([[50, 0], [0, 5]])}]}
>>> ]
>>> 
>>> # Fitting with the interventions
>>> manual_interventions = Intervention(mod=mod)
>>> out_int = manual_interventions.fit(
>>>     y=cp6["sales"], interventions=list_interventions)
>>> dict_smooth = out_int.get("smooth")
>>> data_posterior = dict_smooth.get("posterior")
>>> data_level = data_posterior[data_posterior["parameter"] == "theta_1"].copy()
>>> data_level.head()
```

    ##   parameter        mean  variance  t    ci_lower    ci_upper
    ## 0   theta_1  619.752270  2.150703  1  616.819767  622.684773
    ## 1   theta_1  632.628586  2.620157  2  629.391816  635.865356
    ## 2   theta_1  645.582054  3.487580  3  641.847744  649.316364
    ## 3   theta_1  656.292422  4.831934  4  651.896918  660.687927
    ## 4   theta_1  669.266738  4.427634  5  665.059141  673.474334

``` python
>>> figure(figsize=(12, 8))
>>> plt.plot(cp6["time"], cp6["sales"], "o",
>>>          markersize=4, color="black", fillstyle="none")
>>> # plt.plot(df_fit["f"], color="red")
>>> plt.plot(cp6["time"], data_level["mean"], color="blue")
>>> plt.plot(cp6["time"], data_level["ci_lower"], color="blue",
>>>          linestyle="dashed")
>>> plt.plot(cp6["time"], data_level["ci_upper"], color="blue",
>>>          linestyle="dashed")
>>> plt.grid(linestyle="dotted")
>>> plt.xlabel("Time")
>>> plt.ylabel("Monthly total sales")
>>> plt.show()
```

<img src="examples/figures/README-plot-intervention-3.svg" style="display: block; margin: auto;" />

## Authors

`pybats-detection` was developed by [André
Menezes](https://andrmenezes.github.io/) and [Eduardo
Gabriel](https://www.linkedin.com/in/eduardo-gabriel-433332142/) while
working as Data Scientist at [Murabei Data
Science](https://www.murabei.com/) advised by professor [Hélio
Migon](http://lattes.cnpq.br/7997248190492823) and [André
Baceti](https://br.linkedin.com/in/andre-baceti/pt) .

## License

The `pybats-detection` package is released under the Apache License,
Version 2.0. Please, see file `LICENSE.md`.
