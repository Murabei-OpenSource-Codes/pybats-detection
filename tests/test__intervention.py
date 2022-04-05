"""Unittest for Intervention class."""
import unittest
import numpy as np
from pybats.dglm import dlm
from pybats_detection.random_dlm import RandomDLM
from pybats_detection.intervention import Intervention


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
