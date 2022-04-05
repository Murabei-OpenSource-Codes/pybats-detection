"""Simulate from dynamic linear model with shifts."""
import numpy as np
import pandas as pd
from typing import List


class RandomDLM():
    """A class to simulate random values from DLM."""

    def __init__(self, n: int, V: float, W: List):
        """Simulate random values from Dynamic Linear Models.

        Parameters
        ----------
        n : int
            Sample size.
        V : float
            Observation variance.
        W : List
            List with values for the variance of state parameters.

        """
        self._n = n
        self._V = V
        self._W = W

    def level(self, start_level: float, dict_shift: dict) -> pd.DataFrame:
        """Simulate from dynamic level model.

        Parameters
        ----------
        start_level : float
            Start value for the level parameter.
        dict_shift : dict
            Potential changes at mean and variance of state level parameter.

        Returns
        -------
        pd.DataFrame
            It contains the following columns:
                - `t`: time index.
                - `y`: simulated values.

        """
        mu = np.zeros(self._n)
        mu[0] = start_level

        for t in range(1, self._n):
            omega = 0
            if dict_shift:
                if t in dict_shift["t"]:
                    idx = dict_shift["t"].index(t)
                    level_mean = dict_shift["level_mean_shift"][idx]
                    level_var = dict_shift["level_var_shift"][idx]
                    omega = np.random.normal(
                        level_mean, np.sqrt(level_var * self._W), 1)
            mu[t] = mu[t-1] + omega + np.random.normal(0, np.sqrt(self._W), 1)

        y = np.random.normal(mu, np.sqrt(self._V), self._n)
        t = np.arange(0, len(y)) + 1

        return pd.DataFrame({"t": t, "y": y})

    def growth(self, start_level: float, start_trend: float,
               dict_shift: dict) -> pd.DataFrame:
        """Simulate from dynamic growth model.

        Parameters
        ----------
        start_level : float
            Start value for the level parameter.
        start_trend : float
            Start value for the trend parameter.
        dict_shift : dict
            Potential changes at mean and variance of state level parameter.

        Returns
        -------
        pd.DataFrame
            It contains the following columns:
                - `t`: time index.
                - `y`: simulated values.

        """
        W1 = self._W[0]
        W2 = self._W[1]
        mu = np.zeros(self._n)
        beta = np.zeros(self._n)
        mu[0] = start_level
        beta[0] = start_trend

        for t in range(1, self._n):
            omega = 0
            if dict_shift:
                if t in dict_shift["t"]:
                    idx = dict_shift["t"].index(t)
                    level_mean = dict_shift["mean_shift"][idx]
                    beta[t-1] = beta[t-1] + level_mean
                    mu[t-1] = mu[t-1] + level_mean
            beta[t] = beta[t-1] + np.random.normal(0, np.sqrt(W2), 1)
            mu[t] = (mu[t-1] + beta[t] + omega +
                     np.random.normal(0, np.sqrt(W1), 1))

        y = np.random.normal(mu, np.sqrt(self._V), self._n)
        t = np.arange(0, len(y)) + 1

        return pd.DataFrame({"t": t, "y": y})

    def level_with_covariates(self, X: np.array, start_level: float,
                              start_covariates: List,
                              dict_shift: dict) -> pd.DataFrame:
        """Simulate dynamic level model with covariates.

        Parameters
        ----------
        X : np.array
            A matrix of fixed covariates with dimension p by `n`, where p is
            the number of covariates.
        start_level : float
            Start value for the level parameter.
        start_covariates : List
            Start values for the covariates parameters.
        dict_shift : dict
            Potential changes at mean and variance of state level parameter.
        Returns
        -------
        pd.DataFrame
            It contains the following columns:
                - `t`: time index.
                - `y`: simulated values.
                - `level`: the simulate level values.
                - `x[p]`: the values of covariate p.
                - `theta__x[p]`: the simulate parameter values of covariate p.
        """
        p = X.shape[1]
        W1 = self._W[0]
        W2 = self._W[1:(p+1)]
        mu = np.zeros(self._n)
        betas = np.zeros((self._n, p))
        mu[0] = start_level
        betas[0, :] = start_covariates

        for t in range(1, self._n):
            omega = 0
            if t in dict_shift["t"]:
                idx = dict_shift["t"].index(t)
                level_mean = dict_shift["mean_shift"][idx]
                level_var = dict_shift["var_shift"][idx]
                omega = np.random.normal(
                    level_mean, np.sqrt(level_var * W1), 1)

            betas[t, :] = betas[t-1, :] + np.random.normal(0, np.sqrt(W2), p)
            mu[t] = (start_level + np.sum(X[t - 1, :] * betas[t - 1, :]) +
                     np.random.normal(0, np.sqrt(W1), 1) + omega)

        y = np.random.normal(mu, np.sqrt(self._V), self._n)
        t = np.arange(0, len(y)) + 1

        df_out = pd.DataFrame({"t": t, "y": y, "level": mu})

        for k in range(0, p):
            df_out["x" + str(k+1)] = X[:, k]
            df_out["theta__x" + str(k+1)] = betas[:, k]

        return df_out
