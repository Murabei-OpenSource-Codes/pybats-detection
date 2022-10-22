"""Unittest for Monitoring class."""
import unittest
import pandas as pd
import numpy as np
import copy
from pybats.dglm import dlm
from pybats_detection.random_dlm import RandomDLM
from pybats_detection.monitor import Monitoring
from pybats_detection.loader import load_market_share
from pybats_detection.loader import load_air_passengers


class TestMonitoring(unittest.TestCase):
    """Test Monitoring."""

    def test__unilateral_level(self):
        """Test unilateral level model with automatic monitor."""
        # Generating level data model
        np.random.seed(66)
        rdlm = RandomDLM(n=100, V=10, W=0.6)
        df_simulated = rdlm.level(
            start_level=100,
            dict_shift={"t": [80],
                        "level_mean_shift": [50],
                        "level_var_shift": [1]})

        # Define model (prior mean and variance matrix)
        a = np.array(100)
        R = np.eye(1)
        np.fill_diagonal(R, val=1000)
        mod = dlm(a, R, ntrend=1, deltrend=0.90)

        # Fit with monitoring
        monitor = Monitoring(mod=mod)
        out = monitor.fit(y=df_simulated["y"], bilateral=False)
        self.assertEqual(list(out.keys()), ["filter", "smooth", "model"])

    def test__bilateral_level(self):
        """Test bilateral level model with automatic monitor."""
        # Generating level data model
        np.random.seed(66)
        rdlm = RandomDLM(n=70, V=1, W=0.1)
        df_simulated = rdlm.level(start_level=100,
                                  dict_shift={"t": [30, 50],
                                              "level_mean_shift": [-20, 50],
                                              "level_var_shift": [1, 1]})

        # Define model (prior mean and variance matrix)
        a = np.array(100)
        R = np.eye(1)
        np.fill_diagonal(R, val=1000)
        mod = dlm(a, R, ntrend=1, deltrend=0.90)

        # Fit with monitoring
        monitor = Monitoring(mod=mod)
        out = monitor.fit(y=df_simulated["y"], bilateral=False)
        self.assertEqual(list(out.keys()), ["filter", "smooth", "model"])

    def test__variance_positive_unilateral(self):
        """Test if the variances are positive with unilateral monitor."""
        np.random.seed(66)
        rdlm = RandomDLM(n=70, V=1, W=0.1)
        df_simulated = rdlm.level(start_level=100,
                                  dict_shift={"t": [30, 50],
                                              "level_mean_shift": [-20, 50],
                                              "level_var_shift": [1, 1]})

        # Define model (prior mean and variance matrix)
        a = np.array(100)
        R = np.eye(1)
        np.fill_diagonal(R, val=1000)

        qk_cond = []
        variance_cond = []
        for delta in [0.90, 0.95, 0.975, 1]:
            mod = dlm(a, R, ntrend=1, deltrend=delta)

            # Fit with monitoring
            monitor = Monitoring(mod=mod)
            out = monitor.fit(y=df_simulated["y"], bilateral=False)
            predictive_smooth = out.get('smooth')

            qk = predictive_smooth.get('predictive')['qk']
            variance = predictive_smooth.get('posterior')['variance']
            qk_cond.append((variance > 0.0).all())
            variance_cond.append((variance > 0.0).all())

        self.assertTrue(all(variance_cond), True)
        self.assertTrue(all(qk), True)

    def test__variance_positive_bilateral(self):
        """Test if the variances are positive with bilateral monitor."""
        np.random.seed(66)
        rdlm = RandomDLM(n=70, V=1, W=0.1)
        df_simulated = rdlm.level(start_level=100,
                                  dict_shift={"t": [30, 50],
                                              "level_mean_shift": [-20, 50],
                                              "level_var_shift": [1, 1]})

        # Define model (prior mean and variance matrix)
        a = np.array(100)
        R = np.eye(1)
        np.fill_diagonal(R, val=1000)

        qk_cond = []
        variance_cond = []
        for delta in [0.90, 0.95, 0.975, 1]:
            mod = dlm(a, R, ntrend=1, deltrend=delta)

            # Fit with monitoring
            monitor = Monitoring(mod=mod)
            out = monitor.fit(y=df_simulated["y"], bilateral=True)
            predictive_smooth = out.get('smooth')

            qk = predictive_smooth.get('predictive')['qk']
            variance = predictive_smooth.get('posterior')['variance']
            qk_cond.append((variance > 0.0).all())
            variance_cond.append((variance > 0.0).all())

        self.assertTrue(all(variance_cond), True)
        self.assertTrue(all(qk), True)

    def test__unilateral_level_in_regression(self):
        """Test unilateral level model with automatic monitor in regression."""
        # Generating level data model
        np.random.seed(66)
        X = np.random.normal(0, .1, 100).reshape(-1, 1)
        rdlm = RandomDLM(n=100, V=.1, W=[0.006, .001])
        df_simulated = rdlm.level_with_covariates(
            start_level=100, start_covariates=[-2], X=X,
            dict_shift={"t": [], "mean_shift": [], "var_shift": []})

        # Define model
        a0 = np.array([100, 0, 1])
        R0 = np.eye(3)
        R0[0, 0] = 100
        R0[2, 2] = 10

        mod = dlm(a0=a0, R0=R0, n0=1, s0=.1, delVar=0.98, ntrend=2, nregn=1,
                  delregn=.98, deltrend=0.95)

        # Fit with monitoring
        monitor = Monitoring(mod=mod)
        out = monitor.fit(y=df_simulated["y"], X=df_simulated[["x1"]],
                          bilateral=False, prior_length=20,
                          h=4, tau=0.135)

        self.assertEqual(list(out.keys()), ["filter", "smooth", "model"])

    def test__bilateral_level_in_regression(self):
        """Test bilateral level model with automatic monitor in regression."""
        # Generating level data model
        np.random.seed(66)
        X = np.random.normal(0, .1, 100).reshape(-1, 1)
        rdlm = RandomDLM(n=100, V=.1, W=[0.006, .001])
        df_simulated = rdlm.level_with_covariates(
            start_level=100, start_covariates=[-2], X=X,
            dict_shift={"t": [], "mean_shift": [], "var_shift": []})

        # Define model
        a0 = np.array([100, 0, -1])
        R0 = np.eye(3)
        R0[0, 0] = 100
        R0[2, 2] = 10

        mod = dlm(a0=a0, R0=R0, ntrend=2, nregn=1, delregn=.98, deltrend=0.9)

        # Fit with monitoring
        monitor = Monitoring(mod=mod)
        out = monitor.fit(y=df_simulated.y, X=df_simulated[["x1"]],
                          bilateral=True, prior_length=1, h=4, tau=0.135)
        self.assertEqual(list(out.keys()), ["filter", "smooth", "model"])

    def test__bilateral_level_in_regression_market_share(self):
        """Test bilateral level model with automatic monitor in regression."""
        market_share = load_market_share()
        y = market_share['share']
        X = market_share[['price', 'prom', 'cprom']]
        X = X - X.mean()

        # Define model
        a0 = np.array([42, 0, 0, 0])
        R0 = np.eye(4) * 4.0
        R0[0, 0] = 25

        mod = dlm(a0=a0, R0=R0, ntrend=1, nregn=3, delregn=.90,
                  deltrend=1, delVar=.99)

        # Fit with monitoring
        monitor = Monitoring(mod=mod)
        out = monitor.fit(y=y, X=X,
                          bilateral=True, prior_length=1, h=4, tau=0.135)

        # Measures
        predictive_df = out.get('filter').get('predictive')
        mse = ((predictive_df.y - predictive_df.f)**2).mean()
        mad = np.abs(predictive_df.y - predictive_df.f).mean()

        mse_comparative = np.abs(mse / .056 - 1)
        mad_comparative = np.abs(mad / .185 - 1)

        # Coefs
        mod_ = out.get('model')
        coefs_df = mod_.get_coef()
        signal_lst = list((coefs_df.Mean < 0).values)

        self.assertEqual(list(out.keys()), ["filter", "smooth", "model"])
        self.assertEqual(signal_lst, [False, True, False, True])
        self.assertTrue(mse_comparative < .10)
        self.assertTrue(mad_comparative < .10)

    def test__two_regressors_fk_error(self):
        """Test the smooth predictive error with two regressor."""
        np.random.seed(66)
        mean = (100, 0, 0)
        cov = [[10, .90, .95], [0.90, 1, 0.55], [0.95, 0.55, 1]]

        xy = np.random.multivariate_normal(mean, cov, size=100)
        y = pd.Series(xy[:, 0])
        X = pd.DataFrame(xy[:, [1, 2]])

        # Define model
        a0 = np.array([100, 0, -1, 1])
        R0 = np.eye(4)
        np.fill_diagonal(R0, val=100)

        mod = dlm(a0=a0, R0=R0, ntrend=2, nregn=2, delregn=.98, deltrend=0.95)
        monitor = Monitoring(mod=mod)
        dict_results = monitor.fit(y=y, X=X)

        filter_predictive = dict_results.get("filter").get("predictive")
        smooth_predictive = dict_results.get("smooth").get("predictive")
        smooth_predictive['y'] = filter_predictive['y']

        # MSE
        filter_pe = (filter_predictive['f']/filter_predictive['y'] - 1)
        smooth_pe = (smooth_predictive['fk'] / smooth_predictive['y'] - 1)

        filter_mape = filter_pe.abs().mean()
        smooth_mape = smooth_pe.abs().mean()

        self.assertTrue(filter_mape < 0.05)
        self.assertTrue(smooth_mape < 0.05)
        self.assertTrue(smooth_mape < filter_mape)

    def test__air_passengers_fk_error(self):
        """Test the smooth predictive error on air_passengers data."""
        air_passengers = load_air_passengers()
        a = np.array([112, 0, 0, 0, 0, 0])
        R = np.eye(6)
        np.fill_diagonal(R, val=100)

        mod = dlm(a, R, ntrend=2, deltrend=0.95, delseas=0.98,
                  seasPeriods=[12], seasHarmComponents=[[1, 2]])
        monitor = Monitoring(mod=mod)
        dict_results = monitor.fit(y=air_passengers.total)

        filter_predictive = dict_results.get("filter").get("predictive")
        smooth_predictive = dict_results.get("smooth").get("predictive")
        smooth_predictive['y'] = filter_predictive['y']

        # MSE
        filter_pe = (filter_predictive['f']/filter_predictive['y'] - 1)
        smooth_pe = (smooth_predictive['fk'] / smooth_predictive['y'] - 1)

        filter_mape = filter_pe.abs().mean()
        smooth_mape = smooth_pe.abs().mean()

        self.assertTrue(filter_mape < 0.10)
        self.assertTrue(smooth_mape < 0.05)
        self.assertTrue(smooth_mape < filter_mape)

    def test__monitoring_with_missing_values_unilateral(self):
        """Test the unilateral monitoring with missing values."""
        np.random.seed(66)
        rdlm = RandomDLM(n=70, V=1, W=0.1)
        df_simulated = rdlm.level(start_level=100,
                                  dict_shift={"t": [30, 50],
                                              "level_mean_shift": [-20, 50],
                                              "level_var_shift": [1, 1]})

        # Add y with missing values
        df_simulated['my'] = df_simulated['y']
        df_simulated.loc[[10, 15, 28, 47], 'my'] = np.nan

        # Define model (prior mean and variance matrix)
        a = np.array(100)
        R = np.eye(1)
        np.fill_diagonal(R, val=1000)
        mod = dlm(a, R, ntrend=1, deltrend=0.90)

        # Fit with monitoring
        monitor1 = Monitoring(mod=copy.deepcopy(mod))
        monitor2 = Monitoring(mod=copy.deepcopy(mod))

        out1 = monitor1.fit(y=df_simulated["y"], bilateral=False)
        out2 = monitor2.fit(y=df_simulated["my"], bilateral=False)

        monitor_results1 = out1.get('filter').get('predictive')
        monitor_results2 = out2.get('filter').get('predictive')

        # Interventions times
        query = 'what_detected != "nothing"'
        what_detected_times_1 = monitor_results1.query(query).t.values
        what_detected_times_2 = monitor_results2.query(query).t.values

        # Difference in predictive mean (f)
        f1 = np.abs(monitor_results1.f.values)
        f2 = np.abs(monitor_results2.f.values)
        f_diff = np.abs(np.nanmean(f1/f2) - 1)

        # Compare results
        bool = np.array_equal(what_detected_times_1, what_detected_times_2)
        self.assertTrue(bool)
        self.assertTrue(f_diff < 0.01)

    def test__monitoring_with_missing_values_bilateral(self):
        """Test the unilateral monitoring with missing values."""
        np.random.seed(66)
        rdlm = RandomDLM(n=70, V=1, W=0.1)
        df_simulated = rdlm.level(start_level=100,
                                  dict_shift={"t": [30, 50],
                                              "level_mean_shift": [-20, 50],
                                              "level_var_shift": [1, 1]})

        # Add y with missing values
        df_simulated['my'] = df_simulated['y']
        df_simulated.loc[[10, 15, 30, 50], 'my'] = np.nan

        # Define model (prior mean and variance matrix)
        a = np.array(100)
        R = np.eye(1)
        np.fill_diagonal(R, val=1000)
        mod = dlm(a, R, ntrend=1, deltrend=0.90)

        # Fit with monitoring
        monitor1 = Monitoring(mod=copy.deepcopy(mod))
        monitor2 = Monitoring(mod=copy.deepcopy(mod))

        out1 = monitor1.fit(y=df_simulated["y"], bilateral=True)
        out2 = monitor2.fit(y=df_simulated["my"], bilateral=True)

        monitor_results1 = out1.get('filter').get('predictive')
        monitor_results2 = out2.get('filter').get('predictive')

        # Interventions times
        query = 'what_detected != "nothing"'
        what_detected_times_1 = monitor_results1.query(query).t.values
        what_detected_times_2 = monitor_results2.query(query).t.values - 1

        # Difference in predictive mean (f)
        f1 = np.abs(monitor_results1.f.values)
        f2 = np.abs(monitor_results2.f.values)
        f_diff = np.abs(np.nanmean(f1/f2) - 1)

        # Compare results
        bool = np.array_equal(what_detected_times_1, what_detected_times_2)
        self.assertTrue(bool)
        self.assertTrue(f_diff < 0.01)
