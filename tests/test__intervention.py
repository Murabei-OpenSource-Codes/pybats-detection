"""Unittest for Intervention class."""
import unittest
import pandas as pd
import numpy as np
from pybats.dglm import dlm
from pybats_detection.random_dlm import RandomDLM
from pybats_detection.intervention import Intervention
from pybats_detection.loader import load_market_share
from pybats_detection.loader import load_air_passengers


class TestIntervention(unittest.TestCase):
    """Tests Intervention."""

    def test__variance_intervention(self):
        """Test variance intervention, shifting in observation variance."""
        # Generating level data model
        np.random.seed(66)
        rdlm = RandomDLM(n=50, V=1, W=0.1)
        df_simulated = rdlm.level(
            start_level=100,
            dict_shift={"t": [30, 31, 40, 41],
                        "level_mean_shift": [10, -10, 6, -6],
                        "level_var_shift": [1, 1, 1, 1]})

        # Define model (prior mean and variance matrix)
        a = np.array(100)
        R = np.eye(1)
        np.fill_diagonal(R, val=1000)
        mod = dlm(a0=a, R0=R, ntrend=1, deltrend=0.90)

        # List with the interventions
        list_interventions = [
            {"time_index": 31,
             "which": ["variance"],
             "parameters": [{"v_shift": "ignore"}]},
            {"time_index": 41,
             "which": ["variance"],
             "parameters": [{"v_shift": 10}]}
        ]

        # Perform the filter and smooth with manual interventions
        dlm_intervention = Intervention(mod=mod)
        out = dlm_intervention.fit(
            y=df_simulated["y"], interventions=list_interventions)
        self.assertEqual(list(out.keys()), ["filter", "smooth", "model"])
        self.assertTrue(np.all(out["smooth"]["posterior"]["variance"] > 0))

        dlm_intervention = Intervention(mod=mod, smooth=False)
        out = dlm_intervention.fit(
            y=df_simulated["y"], interventions=list_interventions)
        self.assertEqual(list(out.keys()),
                         ["predictive", "posterior", "model"])

    def test__noise_intervention(self):
        """Test noise intervention, shifting the prior moments."""
        # Generating level data model
        np.random.seed(66)
        rdlm = RandomDLM(n=50, V=1, W=0.1)
        df_simulated = rdlm.level(
            start_level=100,
            dict_shift={"t": [30],
                        "level_mean_shift": [10],
                        "level_var_shift": [1]})

        # Define model (prior mean and variance matrix)
        a = np.array([100, 0])
        R = np.eye(2)
        np.fill_diagonal(R, val=1000)
        mod = dlm(a0=a, R0=R, ntrend=2, deltrend=0.90)

        # List with the interventions
        list_interventions = [
            {"time_index": 31, "which": ["noise"],
             "parameters": [
                {"h_shift": np.array([10, 0]),
                 "H_shift": np.array([[100, 0], [0, 300]])}]
             }]

        # Perform the filter and smooth with manual interventions
        dlm_intervention = Intervention(mod=mod)
        out = dlm_intervention.fit(
            y=df_simulated["y"], interventions=list_interventions)
        self.assertEqual(list(out.keys()), ["filter", "smooth", "model"])
        self.assertTrue(np.all(out["smooth"]["posterior"]["variance"] > 0))

    def test__subjective_intervention(self):
        """Test subjective intervention, replacing the prior moments."""
        # Generating level data model
        np.random.seed(66)
        rdlm = RandomDLM(n=50, V=1, W=0.1)
        df_simulated = rdlm.level(
            start_level=100,
            dict_shift={"t": [30],
                        "level_mean_shift": [10],
                        "level_var_shift": [1]})

        # Define model (prior mean and variance matrix)
        a = np.array([100, 0])
        R = np.eye(2)
        np.fill_diagonal(R, val=1000)
        mod = dlm(a0=a, R0=R, ntrend=2, deltrend=0.90)

        # List with the interventions
        list_interventions = [{
            "time_index": 31, "which": ["subjective"],
            "parameters": [
                {"a_star": np.array([110, 0]),
                 "R_star": np.array([[100, 0], [0, 300]])}]
        }
        ]

        dlm_intervention = Intervention(mod=mod)
        out = dlm_intervention.fit(
            y=df_simulated["y"], interventions=list_interventions)
        self.assertEqual(list(out.keys()), ["filter", "smooth", "model"])
        self.assertTrue(np.all(out["smooth"]["posterior"]["variance"] > 0))

    def test__subjective_intervention_with_regression(self):
        """Test subjective intervention with regressor."""
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

        mod = dlm(a0=a0, R0=R0, n0=1, s0=.1, ntrend=2, nregn=1,
                  delregn=.98, deltrend=0.95)

        # List with the interventions
        list_interventions = [{
            "time_index": 31, "which": ["noise"],
            "parameters": [
                {"h_shift": np.array([110, 0, 0]),
                 "H_shift": np.eye(3)*0.0}]}]

        dlm_intervention = Intervention(mod=mod)
        out = dlm_intervention.fit(
            y=df_simulated["y"], X=df_simulated[["x1"]],
            interventions=list_interventions)
        self.assertEqual(list(out.keys()), ["filter", "smooth", "model"])
        self.assertTrue(np.all(out["smooth"]["posterior"]["variance"] > 0))

    def test__subjective_intervention_with_regression_market_share(self):
        """
        Test subjective intervention with regressor in market share example.
        """
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

        # List with the interventions
        list_interventions = [{
            "time_index": 34, "which": ["variance"],
            "parameters": [
                {"h_shift": np.array([0, 0, 0, 0]),
                 "H_shift": np.eye(4)*0.0}]}]

        dlm_intervention = Intervention(mod=mod)
        out = dlm_intervention.fit(y=y, X=X,
                                   interventions=list_interventions)

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

        # List with the interventions
        list_interventions = [{
            "time_index": 34, "which": ["variance"],
            "parameters": [
                {"h_shift": np.array([0, 0, 0, 0]),
                 "H_shift": np.eye(4)*0.0}]}]

        dlm_intervention = Intervention(mod=mod)
        dict_results = dlm_intervention.fit(y=y, X=X,
                                            interventions=list_interventions)

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
        """Test the smooth predictive error with two regressor."""
        air_passengers = load_air_passengers()
        a0 = np.array([112, 0, 0, 0, 0, 0])
        R0 = np.eye(6)
        np.fill_diagonal(R0, val=100)

        mod = dlm(a0, R0, ntrend=2, deltrend=0.95, delseas=0.98,
                  seasPeriods=[12], seasHarmComponents=[[1, 2]])

        # List with the interventions
        list_interventions = [{
            "time_index": 34, "which": ["variance"],
            "parameters": [
                {"h_shift": np.array([0, 0, 0, 0]),
                 "H_shift": np.eye(4)*0.0}]}]

        dlm_intervention = Intervention(mod=mod)
        dict_results = dlm_intervention.fit(y=air_passengers.total, X=None,
                                            interventions=list_interventions)

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
