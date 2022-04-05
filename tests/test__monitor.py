"""Unittest for Monitoring class."""
import unittest
import numpy as np
from pybats.dglm import dlm
from pybats_detection.random_dlm import RandomDLM
from pybats_detection.monitor import Monitoring


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
                          h=4, tau=0.135, change_var=[100, 1, 1])

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
                          bilateral=True, prior_length=1, h=4, tau=0.135,
                          change_var=[10, 1, 1])
        self.assertEqual(list(out.keys()), ["filter", "smooth", "model"])
