"""Smoothing for Dynamic Linear Models."""
import copy
import pybats
import numpy as np
import pandas as pd
from scipy import stats
from pybats_detection.utils import tidy_parameters


class Smoothing:
    """Retrospective time series analysis.

    Perform retrospective estimation of the historical development of a time
    series mean response function using the smoothed distributions.
    Procedure is done using objects of class `pybats.dglm.dlm`.
    """

    def __init__(self, mod: pybats.dglm.dlm, interval: bool = True,
                 level: float = 0.05):
        """Retrospective time series analysis.

        Perform retrospective estimation of the historical development of a
        time series mean response function using the smoothed distributions.
        Procedure is done using objects of class `pybats.dglm.dlm`.

        Parameters
        ----------
        mod : pybats.dglm.dlm
            An object of class `pybats.dglm.dlm` with the DLM updated.
        interval : bool
            Indicate if the credible interval was calculated.
        level : float
            The probability level used to compute the credible interval.

        Attributes
        ----------
        _mod : pybats.dglm.dlm
            An object of class `pybats.dglm.dlm` with the DLM updated.
        _interval : bool
            Indicate if the credible interval was calculated.
        _level : float
            The probability level used to compute the credible interval.

        """
        self._mod = mod
        self._interval = interval
        self._level = level

    def fit(self, y: pd.Series, X: pd.DataFrame = None,
            dict_state_parms: dict = None):
        """Perform the backward smoother.

        That is, performs retrospective estimation for the state space
        parameters calculating the smoothed moments of the one-step ahead
        predictive distribution using the moments in `dict_state_parms`.
        If `dict_state_parms` is None then first compute the forward filter on
        state parameters and predictive distributions.

        Parameters
        ----------
        y : pd.Series
            Observed values of time series.
        X : pd.DataFrame
            Observed values of fixed covariates with dimension `p` by `n`,
            where `p` is the number of covariates.
        dict_state_parms : dict
            A dictionary contains the historical posterior and prior momments
            with the following keys:
            ```
            dict_state_parms = {
                "prior": {"a": [], "R": []},
                "posterior": {"m": [], "C": [], "df": [], "s": []}
            }
            ```
            where `m` and `C` are the posterior mean and covariance matrix,
            respectively, 'df' is the degree of freedom, `s` is the
            observational variance and `a` and `R` are the prior mean and
            covariance matrix, respectively. If `dict_state_parms` is `None`
            the forward filtering is perform before in order to obtain the
            posterior and prior moments values.

        Returns
        -------
        dict. It contains the following keys:
            - `model`: the updated pybats.dglm.dlm object.
            - `smooth`: dictionary with:
                - `posterior`: pd.DataFrame with the smooth posterior moments.
                - `predictive`: pd.DataFrame with the smooth one-step ahead
                predictive moments.

            If `dict_state_parms` is None, then the dict also contains:
            - `filter`: dictionary with:
                - `posterior`: pd.DataFrame with the filtering posterior
                moments.
                - `predictive`: pd.DataFrame with the one-step ahead predictive
                moments.

        References
        ----------
        [1] West, M.; Harrison, J. Bayesian Forecasting and Dynamic Models.
        Springer, 1997.

        [2] Prado, R.; West, M. Time Series Modeling, Computation, and
        Inference. CRC Press, 2010.
        """
        n = len(y)
        # Create a None vector when there are no regression covariates
        if X is None:
            X = pd.DataFrame(np.array([None]*(n+1)).reshape(-1, 1))

        self._pd_y = y.copy()
        self._pd_X = X.copy()
        self._dict_state_parms = dict_state_parms

        if self._dict_state_parms:
            tmp = self._backward_smoother(
                dict_state_parms=self._dict_state_parms)
            out = {"predictive": tmp[0], "posterior": tmp[1]}
        else:
            tmp_ff = self._forward_filter()
            tmp_bs = self._backward_smoother(dict_state_parms=tmp_ff[2])
            out = {
                "model": tmp_ff[3],
                "filter": {"predictive": tmp_ff[0], "posterior": tmp_ff[1]},
                "smooth": {"predictive": tmp_bs[0], "posterior": tmp_bs[1]}}

        return out

    def _forward_filter(self):
        """Perform forward filtering.

        That is, obtain the moments of the
        the one-step ahead predictive distribution and state space posterior
        distribution.

        Returns
        -------
            List: It contains the following components:
            - `df_predictive`: pd.DataFrame with the moments of predictive
            distribution

            - `df_posterior`: pd.DataFrame with the moments of posterior state
            space distribution.

            - `dict_state_parms`: dictionary with the posterior (m and C) and
            prior (a and R) moments for the state space parameters along time.

        """
        # Data and model
        pd_y = self._pd_y.copy()
        pd_X = self._pd_X.copy()
        c_mod = copy.deepcopy(self._mod)
        n = len(pd_y)

        # Dictionaries to keep state and predictive parameters
        dict_predictive = {
            "t": [], "y": [], "f": [], "q": [], "df": [], "s": []}
        dict_state_parms = {
            "prior": {"a": [], "R": []},
            "posterior": {"m": [], "C": [], "df": [], "s": []}
        }

        # Perform the filtering and keep the paramters
        for t in range(0, n):
            yt = pd_y.values[t]
            Xt = pd_X.values[t, :]

            # Get mean and variance one step ahead of forecast distribution
            ft, qt = c_mod.forecast_marginal(k=1, X=Xt, state_mean_var=True)
            ft = ft[0][0]
            qt = qt[0][0]

            # Saving prior state parameters
            dict_state_parms["prior"]["a"].append(c_mod.a)
            dict_state_parms["prior"]["R"].append(c_mod.R)
            dict_state_parms["posterior"]["df"].append(c_mod.n)
            dict_state_parms["posterior"]["s"].append(c_mod.s)

            # Update model
            c_mod.update(y=yt, X=Xt)

            # Saving posterior state parameters
            dict_state_parms["posterior"]["m"].append(c_mod.m)
            dict_state_parms["posterior"]["C"].append(c_mod.C)

            # Saving parameters
            dict_predictive["t"].append(t + 1)
            dict_predictive["y"].append(yt)
            dict_predictive["f"].append(ft)
            dict_predictive["q"].append(qt)
            dict_predictive["df"].append(c_mod.n - 1)
            dict_predictive["s"].append(c_mod.s[0][0])
            # end of loop

        # Organize the predictive parameters in DataFrame
        df_predictive = pd.DataFrame(dict_predictive)

        # Organize the posterior parameters in DataFrame
        df_posterior = tidy_parameters(
            dict_parameters=dict_state_parms["posterior"],
            entry_m="m", entry_v="C",
            names_parameters=list(c_mod.get_coef().index),
            index_seas_parameters=c_mod.iseas,
            F=c_mod.F)
        n_parms = len(df_posterior["parameter"].unique())
        t_index = np.arange(0, len(df_posterior) / n_parms) + 1
        df_posterior["t"] = np.repeat(t_index, n_parms)
        df_posterior["t"] = df_posterior["t"].astype(int)
        df_posterior = df_posterior.merge(
            df_predictive[['t', 'df']], on="t", how="left",
            validate="many_to_one")

        df_posterior = df_posterior[
            ["t", "parameter", "mean", "variance", "df"]].copy()
        df_posterior = df_posterior.sort_values(
            ["parameter", "t"]).reset_index(drop=True)

        if self._interval:
            df_predictive["ci_lower"] = stats.t.ppf(
                q=self._level/2,
                df=df_predictive["df"].values,
                loc=df_predictive["f"].values,
                scale=np.sqrt(df_predictive["q"].values))
            df_predictive["ci_upper"] = stats.t.ppf(
                q=1-self._level/2,
                df=df_predictive["df"].values,
                loc=df_predictive["f"].values,
                scale=np.sqrt(df_predictive["q"].values))
            df_posterior["ci_lower"] = stats.t.ppf(
                q=self._level/2,
                df=df_posterior["df"].values,
                loc=df_posterior["mean"].values,
                scale=np.sqrt(df_posterior["variance"].values))
            df_posterior["ci_upper"] = stats.t.ppf(
                q=1-self._level/2,
                df=df_posterior["df"].values,
                loc=df_posterior["mean"].values,
                scale=np.sqrt(df_posterior["variance"].values))

        self._dict_predictive = dict_predictive
        return df_predictive, df_posterior, dict_state_parms, c_mod

    def _backward_smoother(self, dict_state_parms: dict):
        """Perform backward smoother.

        That is, obtain the smoothing moments of the one-step ahead predictive
        distribution and state space posterior distribution.

        Parameters
        ----------
        dict_state_parms : dict
            dictionary with the posterior (m and C) and prior (a and R) moments
            for the state space parameters along time.

        Returns
        -------
        List: It contains the following components:
            - `df_predictive_smooth`: pd.DataFrame with the smoothing moments
            of predictive distribution.

            - `df_posterior_smooth`: pd.DataFrame with the smoothing moments
            of posterior state space distribution.
        """
        # Initialize the model components and posterior/prior parameters
        T_end = len(self._pd_y)
        F = self._mod.F
        G = self._mod.G
        R = dict_state_parms["prior"]["R"]
        a = dict_state_parms["prior"]["a"]
        C = dict_state_parms["posterior"]["C"]
        m = dict_state_parms["posterior"]["m"]
        # s = dict_state_parms["posterior"]["s"]
        df = dict_state_parms["posterior"]["df"]

        # Dictionaty to save predictive and posterior parameters
        ak = m[T_end-1]
        Rk = C[T_end-1]
        fk = F.T @ ak
        qk = F.T @ Rk @ F
        dict_smooth_parms = {
            "t": [T_end], "ak": [ak], "Rk": [Rk], "fk": [fk[0][0]],
            "qk": [qk[0][0]], "df": np.flip(df)}

        # Perform smoothing
        for k in range(1, T_end):
            # B_{t-k}
            B_t_k = C[T_end-k-1] @ G.T @ np.linalg.pinv(R[T_end-k])

            # a_t(-k) and R_t(-k)
            ak = m[T_end-k-1] + B_t_k @ (ak - a[T_end-k])
            Rk = C[T_end-k-1] + B_t_k @ (Rk - R[T_end-k]) @ B_t_k.T

            # f_t(-k) and q_t(-k)
            fk = F.T @ ak
            # qk = (s[T_end - k] / s[T_end - k - 1]) * F.T @ Rk @ F
            qk = F.T @ Rk @ F

            # Saving parameters
            dict_smooth_parms["ak"].append(ak)
            dict_smooth_parms["Rk"].append(Rk)
            dict_smooth_parms["fk"].append(fk[0][0])
            dict_smooth_parms["qk"].append(qk[0][0])
            dict_smooth_parms["t"].append(T_end-k)

        # Organize the predictive smooth parameters
        dict_filtered = {key: dict_smooth_parms[key] for key in (
            dict_smooth_parms.keys() & {"t", "fk", "qk", "df"})}
        df_predictive_smooth = pd.DataFrame(dict_filtered)

        # Organize the posterior smooth parameters in pd.DataFrame
        df_posterior_smooth = tidy_parameters(
            dict_parameters=dict_smooth_parms,
            entry_m="ak", entry_v="Rk",
            names_parameters=list(self._mod.get_coef().index),
            index_seas_parameters=self._mod.iseas,
            F=self._mod.F)
        n_parms = len(df_posterior_smooth["parameter"].unique())
        df_posterior_smooth.reset_index(inplace=True)
        df_posterior_smooth = df_posterior_smooth.sort_values(
            ['parameter', 'index'])
        df_posterior_smooth["t"] = np.tile(dict_smooth_parms["t"], n_parms)

        df_posterior_smooth = df_posterior_smooth.merge(
            df_predictive_smooth[['t', 'df']], on="t", how="left",
            validate="many_to_one")
        df_posterior_smooth = df_posterior_smooth[
            ["t", "parameter", "mean", "variance", "df"]].copy()

        # Arrange DataFrames by time
        df_predictive_smooth = df_predictive_smooth.sort_values(
            ["t"]).reset_index(drop=True)
        df_posterior_smooth = df_posterior_smooth.sort_values(
            ["parameter", "t"]).reset_index(drop=True)

        if self._interval:
            df_predictive_smooth["ci_lower"] = stats.t.ppf(
                q=self._level/2, df=df_predictive_smooth["df"].values[-1],
                loc=df_predictive_smooth["fk"].values,
                scale=np.sqrt(df_predictive_smooth["qk"].values))
            df_predictive_smooth["ci_upper"] = stats.t.ppf(
                q=1-self._level/2,
                df=df_predictive_smooth["df"].values[-1],
                loc=df_predictive_smooth["fk"].values,
                scale=np.sqrt(df_predictive_smooth["qk"].values))

            df_posterior_smooth["ci_lower"] = stats.t.ppf(
                q=self._level/2, df=df_posterior_smooth["df"].values[-1],
                loc=df_posterior_smooth["mean"].values,
                scale=np.sqrt(df_posterior_smooth["variance"].values))
            df_posterior_smooth["ci_upper"] = stats.t.ppf(
                q=1-self._level/2,
                df=df_posterior_smooth["df"].values[-1],
                loc=df_posterior_smooth["mean"].values,
                scale=np.sqrt(df_posterior_smooth["variance"].values))

        self._dict_smooth_parms = dict_smooth_parms
        return df_predictive_smooth, df_posterior_smooth
