"""Intervention analysis in Bayesian Dynamic Linear Models."""
import copy
import pybats
import numpy as np
import pandas as pd
from scipy import stats
from typing import List
from pybats_detection.smooth import Smoothing
from pybats_detection.utils import tidy_parameters


class Intervention:
    """Bayesian Intervention analysis.

    Perform intervention analysis on Dynamic Linear Models from objects of
    class `pybats.dglm.dlm`.
    """

    def __init__(self, mod: pybats.dglm.dlm, smooth: bool = True,
                 interval: bool = True, level: float = 0.05):
        """Bayesian Intervention analysis.

        Perform intervention analysis on Dynamic Linear Models from objects of
        class `pybats.dglm.dlm`.

        Parameters
        ----------
        mod : pybats.dglm.dlm
            An object of class `pybats.dglm.dlm` with the defined DLM.
        smooth : bool
            Should compute the smoothing moments?
        interval : bool
            Should credibile interval be calculated?
        level : float
            A number between 0 and 1 indicating the probability level of the
            credible interval.

        Attributes
        ----------
        _mod : pybats.dglm.dlm
            An object of class `pybats.dglm.dlm` with the DLM updated.
        _interval : bool
            Indicate if the credible interval was calculated.
        _level : double
            The probability level used to compute the credible interval.

        """
        self._mod = copy.deepcopy(mod)
        self._interval = interval
        self._level = level
        self._smooth = smooth

    def fit(self, y: pd.Series, X: pd.DataFrame = None,
            interventions: List = []):
        """Perform the fit with manual intervention.

        Filtering and smoothing distribution with manual intervention on
        objects of class `pybats.dglm.dlm`.

        Parameters
        ----------
        y : pd.Series
            Observed values of time series.
        X : pd.DataFrame
            Observed values of fixed covariates with dimension `p` by `n`,
            where `p` is the number of covariates.
        interventions : List
            List of dictionaries with intervention parameters. The dictionary
            must have the following keys:
                - `time_index`: the time index when the intervention must be
                performed;
                - `which`: the type of intervetion. They are: `variance`,
                `noise`, and `subjective`.
                - `parameters`: the parameters and theirs values to change.

                The `variance` intervention type changes the observation
                variance, s_t. The only parameter is `v_shift`, which the
                possible values are: a character named `ignore` meaning simply
                treating y_t as an outlier and ignore it (the same as
                increase to infinity the observation variance) or a numeric
                value, v_t, to sum the current value of s_t.

                Other intervention type is specified by `noise` option, which
                is the additional evolution noise intervention.
                Its parameters are `h_shift` and `H_shift` that reflect,
                respectively, a shit on the prior mean and an inflation in
                uncertainty through the prior covariance matrix.
                `h_shift` and `H_shift` must be a np.ndarray vector and
                matrix with the corresponding same dimension of the state
                parameter, respectively.
                If any of the parameters, `h_shift` and `H_shift`, are not
                specify, then the current values of mean vector or covariance
                matrix are not alter.

                When the `subjective` option is chosen the arbitrary subjective
                intervention is perform. The parameters are 'a_star' and
                `R_star` that change the current prior mean vector and
                covariance matrix from those one specified in the dictionary.
                As well as in `noise` option `a_star` and `R_star` follows the
                same structure of 'h_shift' and `H_shift`, respectively.

                The difference between `noise` and `subjective`, is that the
                noise increase/decrease the mean or variance of the state
                parameters, while the `subjective` change the prior mean vector
                and variance matrix from those one of `a_star` and `R_star`.

        Returns
        -------
        dict. It contains the following entries:
            - `model`: the updated pybats.dglm.dlm object.

            - `filter`: a dictionary with:
                - `posterior`: pd.DataFrame with the filtering posterior
                moments.
                - `predictive`: pd.DataFrame with the one-step ahead predictive
                moments.

            If `smooth` is True, then the dict also contains:
            - `smooth`: a dictionary with:
                - `posterior`: pd.DataFrame with the smooth posterior moments.
                - `predictive`: pd.DataFrame with the smooth one-step ahead
                predictive moments.

        References
        ----------
        [1] West, M.; Harrison, J. Bayesian Forecasting and Dynamic Models.
        Springer, 1997.

        [2] West, M., Harrison, J., 1989. Subjective intervention in formal
        models. Journal of Forecasting 8, 33–53.
        """
        n = len(y)
        # Create a None vector when there are no regression covariates
        if X is None:
            X = pd.DataFrame(np.array([None]*(n+1)).reshape(-1, 1))

        # Data and model
        pd_y = y.copy()
        pd_X = X.copy()

        # Dictionaries to keep state and predictive parameters
        dict_predictive = {
            "t": [], "intervention_type": [], "y": [], "f": [], "q": [],
            "df": [], "s": []
        }
        dict_state_parms = {
            "prior": {"a": [], "R": []},
            "posterior": {"m": [], "C": [], "df": [], "s": []}
        }
        # Perform the fit with intervention
        for t in range(0, n):
            which_intervention = "nothing"
            self._y_cur = pd_y.values[t]
            Xt = pd_X.values[t, :]
            time_interventions = list(
                map(lambda x: x["time_index"] - 1, interventions))
            if t in time_interventions:
                idx = time_interventions.index(t)
                which = interventions[idx]["which"]
                parms = interventions[idx]["parameters"]
                if len(parms) != len(which):
                    print("The length of 'which' must be equal to the \
                    length of parameters.")
                    break
                self._make_intervention(which=which, parms=parms)
                which_intervention = ", ".join(which)

            # Get mean and variance one step ahead of forecast distribution
            ft, qt = self._mod.forecast_marginal(
                k=1, X=Xt, state_mean_var=True)
            ft = ft[0][0]
            qt = qt[0][0]

            dict_state_parms["prior"]["a"].append(self._mod.a)
            dict_state_parms["prior"]["R"].append(self._mod.R)
            dict_state_parms["posterior"]["df"].append(self._mod.n)
            dict_state_parms["posterior"]["s"].append(self._mod.s)

            # Update model
            self._mod.update(y=self._y_cur, X=Xt)

            # Saving state parameters
            dict_state_parms["posterior"]["m"].append(self._mod.m)
            dict_state_parms["posterior"]["C"].append(self._mod.C)

            # Saving parameters
            dict_predictive["t"].append(t + 1)
            dict_predictive["y"].append(pd_y.values[t])
            dict_predictive["f"].append(ft)
            dict_predictive["q"].append(qt)
            dict_predictive["df"].append(self._mod.n - 1)
            dict_predictive["s"].append(self._mod.s[0][0])
            dict_predictive["intervention_type"].append(which_intervention)
            # end of loop

        # Predictive
        df_predictive = pd.DataFrame(dict_predictive)

        # Posterior parameters
        df_posterior = tidy_parameters(
            dict_parameters=dict_state_parms["posterior"],
            entry_m="m", entry_v="C",
            names_parameters=list(self._mod.get_coef().index),
            index_seas_parameters=self._mod.iseas,
            F=self._mod.F)
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
                q=self._level/2, df=df_predictive["df"].values,
                loc=df_predictive["f"].values,
                scale=np.sqrt(df_predictive["q"].values))
            df_predictive["ci_upper"] = stats.t.ppf(
                q=1-self._level/2, df=df_predictive["df"].values,
                loc=df_predictive["f"].values,
                scale=np.sqrt(df_predictive["q"].values))

            df_posterior["ci_lower"] = stats.t.ppf(
                q=self._level/2, df=df_posterior["df"].values,
                loc=df_posterior["mean"].values,
                scale=np.sqrt(df_posterior["variance"].values))
            df_posterior["ci_upper"] = stats.t.ppf(
                q=1-self._level/2,
                df=df_posterior["df"].values,
                loc=df_posterior["mean"].values,
                scale=np.sqrt(df_posterior["variance"].values))

        out = {"predictive": df_predictive, "posterior": df_posterior}
        if self._smooth:
            smooth_class = Smoothing(
                mod=self._mod, interval=self._interval, level=self._level)
            dict_smooth = smooth_class.fit(
                y=y, dict_state_parms=dict_state_parms)
            out = {"filter": out, "smooth": dict_smooth}

        out["model"] = self._mod

        return out

    def _make_intervention(self, which, parms):
        for j in range(0, len(which)):
            make = self._get_intervention(which=which[j])
            make(parms=parms[j])

    def _get_intervention(self, which):
        switcher = {
            "variance": self._variance_intervention,
            "noise": self._noise_intervention,
            "subjective": self._subjective_intervention
        }
        return switcher[which]

    def _variance_intervention(self, parms):
        v_shift = parms.get("v_shift")
        if v_shift == "ignore":
            self._y_cur = None
        if isinstance(v_shift, (float, int)):
            self._mod.s = self._mod.s + v_shift

    def _noise_intervention(self, parms):
        h_shift = parms.get("h_shift")
        H_shift = parms.get("H_shift")
        if h_shift is not None:
            self._mod.a = self._mod.a + h_shift.reshape(h_shift.shape[0], 1)
        if H_shift is not None:
            self._mod.R = self._mod.R + H_shift

    def _subjective_intervention(self, parms):
        a_star = parms.get("a_star")
        R_star = parms.get("R_star")
        if a_star is not None:
            self._mod.a = a_star.reshape(a_star.shape[0], 1)
        if R_star is not None:
            self._mod.R = R_star
