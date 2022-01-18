"""Unittest for AutomaticMonitoring class."""
import unittest
import numpy as np
from pybats.dglm import dlm
from pybats_detection.random_dlm import RandomDLM
from pybats_detection.monitor import AutomaticMonitoring


class TestAutomaticMonitoring(unittest.TestCase):
    """Test AutomaticMonitoring."""

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
        monitor = AutomaticMonitoring(mod=mod, bilateral=False)
        out = monitor.fit(y=df_simulated["y"])
        self.assertEqual(list(out.keys()), ["filter", "smooth"])

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
        monitor = AutomaticMonitoring(mod=mod, bilateral=False)
        out = monitor.fit(y=df_simulated["y"])
        self.assertEqual(list(out.keys()), ["filter", "smooth"])

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
            monitor = AutomaticMonitoring(mod=mod, bilateral=False)
            out = monitor.fit(y=df_simulated["y"])
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
            monitor = AutomaticMonitoring(mod=mod, bilateral=True)
            out = monitor.fit(y=df_simulated["y"])
            predictive_smooth = out.get('smooth')

            qk = predictive_smooth.get('predictive')['qk']
            variance = predictive_smooth.get('posterior')['variance']
            qk_cond.append((variance > 0.0).all())
            variance_cond.append((variance > 0.0).all())

        self.assertTrue(all(variance_cond), True)
        self.assertTrue(all(qk), True)
