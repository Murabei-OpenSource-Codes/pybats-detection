"""Unit tests for Smooth class."""
import pandas as pd
import unittest
import numpy as np
from pybats.dglm import dlm
from pybats_detection.smooth import Smoothing
from pybats_detection.loader import load_air_passengers
from pybats_detection.random_dlm import RandomDLM
from pybats_detection.loader import load_market_share

# Generating level data model
np.random.seed(66)
rdlm = RandomDLM(n=100, V=1, W=1)
df_simulated = rdlm.level(
    start_level=100,
    dict_shift={"t": [80],
                "level_mean_shift": [0],
                "level_var_shift": [0]})
y = df_simulated["y"]

# Define model (prior mean and variance matrix)
a = np.array([100, 0])
R = np.eye(2)
np.fill_diagonal(R, val=100)


class TestSmoothing(unittest.TestCase):
    """Unit tests for Smooth class."""

    def test__variance_positive(self):
        """Test if the variances are positive."""
        qk_cond = []
        variance_cond = []

        for delta in [0.90, 0.95, 0.975, 1]:
            mod = dlm(a, R, ntrend=2, deltrend=delta)
            smooth = Smoothing(mod=mod)
            predictive_smooth = smooth.fit(y=y).get('smooth')

            qk = predictive_smooth.get('predictive')['qk']
            variance = predictive_smooth.get('posterior')['variance']
            qk_cond.append((variance > 0.0).all())
            variance_cond.append((variance > 0.0).all())

        self.assertTrue(all(variance_cond), True)
        self.assertTrue(all(qk), True)

    def test__interval_length(self):
        """Test length of credibility interval."""
        len_ci_cond = []
        for delta in [0.90, 0.95, 0.975, 1]:
            mod = dlm(a, R, ntrend=2, deltrend=delta)
            smooth = Smoothing(mod=mod)
            predictive_smooth = smooth.fit(y=y).get('smooth').get('predictive')

            fk = predictive_smooth.loc[:, 'fk']
            ci_upper = predictive_smooth.loc[:, 'ci_upper']
            half_length = ci_upper - fk
            len_ci_cond.append(half_length.max() < 2.50)

        self.assertTrue(all(len_ci_cond), True)

    def test__fk_error(self):
        """Test error for predictive mean."""
        mod = dlm(a, R, ntrend=2, deltrend=0.90)
        smooth = Smoothing(mod=mod)
        predictive_smooth = smooth.fit(y=y).get('smooth').get('predictive')

        fk = predictive_smooth.loc[:, 'fk']
        error = np.sum(fk - y)**2
        self.assertTrue((error < 0.5), True)

    def test__deltrend_qk_relation(self):
        """Test if predictive variance increases with higher delta's."""
        sum_qk = 30
        sum_qk_cond = []
        for delta in [0.90, 0.95, 0.975, 1]:
            mod = dlm(a, R, ntrend=2, deltrend=delta)
            smooth = Smoothing(mod=mod)
            predictive_smooth = smooth.fit(y=y).get('smooth').get('predictive')

            sum_qk_cond.append(predictive_smooth['qk'].sum() < sum_qk)
            sum_qk = predictive_smooth['qk'].sum()

        self.assertTrue(all(sum_qk_cond), True)

    def test__air_passengers(self):
        """Test the smooth on air_passengers data."""
        air_passengers = load_air_passengers()
        a = np.array([112, 0, 0, 0, 0, 0])
        R = np.eye(6)
        np.fill_diagonal(R, val=100)
        mod = dlm(a, R, ntrend=2, deltrend=0.95, delseas=0.98,
                  seasPeriods=[12], seasHarmComponents=[[1, 2]])
        smooth = Smoothing(mod=mod)
        dict_results = smooth.fit(y=air_passengers["total"])
        data_posterior = dict_results.get("filter").get("posterior")
        data_posterior_smooth = dict_results.get("smooth").get("posterior")
        self.assertTrue((data_posterior["parameter"] == "Sum Seas 1").any())

        expected_cols = ["t", "parameter", "mean", "variance", "df",
                         "ci_lower", "ci_upper"]
        self.assertEqual(list(data_posterior.columns), expected_cols)
        self.assertEqual(list(data_posterior_smooth.columns), expected_cols)

    def test__one_regressor(self):
        """Test the smooth with one regressor."""
        # Generating level data model
        np.random.seed(66)
        X = np.random.normal(0, .1, 100).reshape(-1, 1)
        rdlm = RandomDLM(n=100, V=.1, W=[0.006, .001])
        df_simulated = rdlm.level_with_covariates(
            start_level=100, start_covariates=[-2], X=X,
            dict_shift={"t": [30], "mean_shift": [10], "var_shift": [1]})

        # Define model
        a0 = np.array([100, 0, -1])
        R0 = np.eye(3)
        R0[0, 0] = 100
        R0[2, 2] = 10

        mod = dlm(a0=a0, R0=R0, ntrend=2, nregn=1, delregn=.98, deltrend=0.95)

        smooth = Smoothing(mod=mod)
        dict_results = smooth.fit(y=df_simulated["y"], X=df_simulated[["x1"]])
        filter_posterior = dict_results.get("filter").get("posterior")
        smooth_posterior = dict_results.get("smooth").get("posterior")

        t100_filter_posterior = filter_posterior[filter_posterior.t == 100]
        t100_smooth_posterior = smooth_posterior[smooth_posterior.t == 100]

        t100_filter_posterior.reset_index(inplace=True, drop=True)
        t100_smooth_posterior.reset_index(inplace=True, drop=True)

        self.assertTrue(t100_filter_posterior.equals(t100_smooth_posterior))

    def test__two_regressor(self):
        """Test the smooth with two regressor."""
        # Generating level data model
        np.random.seed(66)
        X = np.random.normal(0, .1, 100).reshape(-1, 1)
        rdlm = RandomDLM(n=100, V=.1, W=[0.006, .001])
        df_simulated = rdlm.level_with_covariates(
            start_level=100, start_covariates=[-2], X=X,
            dict_shift={"t": [30], "mean_shift": [10], "var_shift": [1]})
        df_simulated["x2"] = df_simulated["x1"] + np.random.normal(2, 1, 100)

        # Define model
        a0 = np.array([100, 0, -1, 1])
        R0 = np.eye(4)
        np.fill_diagonal(R0, val=100)

        mod = dlm(a0=a0, R0=R0, ntrend=2, nregn=2, delregn=.98, deltrend=0.95)

        smooth = Smoothing(mod=mod)
        dict_results = smooth.fit(y=df_simulated["y"],
                                  X=df_simulated[["x1", "x2"]])
        filter_posterior = dict_results.get("filter").get("posterior")
        smooth_posterior = dict_results.get("smooth").get("posterior")

        t100_filter_posterior = filter_posterior[filter_posterior.t == 100]
        t100_smooth_posterior = smooth_posterior[smooth_posterior.t == 100]

        t100_filter_posterior.reset_index(inplace=True, drop=True)
        t100_smooth_posterior.reset_index(inplace=True, drop=True)

        self.assertTrue(t100_filter_posterior.equals(t100_smooth_posterior))

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
        monitor = Smoothing(mod=mod)
        out = monitor.fit(y=y, X=X)

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
        smooth = Smoothing(mod=mod)
        dict_results = smooth.fit(y=y, X=X)

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
