"""Unit tests for Smooth class."""
import unittest
import numpy as np
from pybats.dglm import dlm
from pybats_detection.random_dlm import RandomDLM
from pybats_detection.smooth import Smoothing

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
np.fill_diagonal(R, val=1)


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
