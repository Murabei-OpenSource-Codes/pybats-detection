"""Monitoring in Bayesian Dynamic Linear Models."""
import copy
import pybats
import numpy as np
import pandas as pd
from scipy import stats
from pybats_detection.smooth import Smoothing
from pybats_detection.utils import tidy_parameters


class Monitoring:
    """Bayesian Monitoring analysis.

    Perform automatic monitoring analysis on Dynamic Linear Models from objects
    of class `pybats.dglm.dlm`.
    """

    def __init__(self, mod: pybats.dglm.dlm, smooth: bool = True,
                 interval: bool = True, level: float = 0.05):
        """Automatic Monitoring Analysis.

        Perform automatic monitoring analysis on Dynamic Linear Models from
        objects of class `pybats.dglm.dlm`.

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
        _smooth : bool
            Indicate if the smooth distribution was calculated.
        _interval : bool
            Indicate if the credible interval was calculated.
        _level : double
            The probability level used to compute the credible interval.

        """
        self._mod = mod
        self._smooth = smooth
        self._interval = interval
        self._level = level

    def fit(self, y: pd.Series, X: pd.DataFrame = None, prior_length: int = 10,
            bilateral: bool = False, h: int = 4, tau: float = 0.135,
            discount_factors: dict = {
                "trend": 0.10, "seasonal": 0.90, "reg": 0.98},
            distr_fam: str = "normal",
            distr_type: str = "location", verbose: bool = True):
        """Perform the fit with automatic monitoring.

        Filtering and smoothing distribution with automatic monitoring on
        objects of class `pybats.dglm.dlm`.

        Parameters
        ----------
        y : pd.Series
            Observed values of time series.
        X : pd.DataFrame
            Observed values of fixed covariates with dimension `p` by `n`,
            where `p` is the number of covariates.
        prior_length : int
            Number of observation with the monitor off.
        bilateral : bool
            Should bilateral monitoring be performed?
        h : double
            Value of change in the scale or location of the predictive
            distribution.
        tau : double
            Threshold to compare the Bayes factor.
        discount_factors : dict
            Discount factors values to increase the uncertainty about
            state space parameter, when the monitor detects potential outlier
            or parametric change.

            The dictionary should contain values with the following keys
            representing the model blocks:
                - `trend`: level and growth
                - `seasonal`: seasonality
                - `reg`: regressors
        distr_fam : str
            Bayes factor distribution family. It could be "normal" or
            "tstudent".
        distr_type : str
            Alternative distribution use to compute the Bayes factor.
            It could be "location" to detects change in the location of the
            distribution or "scale" to dectecs changes in the scale/dispersion
            of the predictive distribution.
        verbose : bool
            If `True` displays the detection.

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

        [2] West, M., Harrison, P.J., 1986. Monitoring and adaptation in
        bayesian forecasting models. Journal of the American Statistical
        Association 81, 741–750.

        """
        n = len(y)
        # Create a None vector when there are no regression covariates
        if X is None:
            X = pd.DataFrame(np.array([None]*(n+1)).reshape(-1, 1))

        self._pd_y = y.copy()
        self._pd_X = X.copy()
        self._verbose = verbose
        self._prior_length = prior_length

        # Check discount factor parameters
        discount_factors["level"] = (
            self._mod.deltrend if discount_factors.get("level") is None
            else discount_factors["level"])
        discount_factors["seasonal"] = (
            self._mod.delseas if discount_factors.get("seasonal") is None
            else discount_factors["seasonal"])
        discount_factors["reg"] = (
            self._mod.delregn if discount_factors.get("reg") is None
            else discount_factors["reg"])
        self._discount_intervention = discount_factors

        if type == "scale":
            return self._fit_unilateral(h=h, tau=tau,
                                        distr_fam=distr_fam,
                                        distr_type=distr_type)

        if bilateral:
            return self._fit_bilateral(h=h, tau=tau,
                                       distr_fam=distr_fam,
                                       distr_type=distr_type)
        else:
            return self._fit_unilateral(h=h, tau=tau,
                                        distr_fam=distr_fam,
                                        distr_type=distr_type)

    def _fit_unilateral(self, h, tau, distr_fam, distr_type):
        mod = copy.deepcopy(self._mod)
        pd_y = self._pd_y.copy()
        pd_X = self._pd_X.copy()
        n = len(pd_y)

        # Dictionaries to keep state and predictive parameters
        dict_state_parms = {
            "prior": {"a": [], "R": []},
            "posterior": {"m": [], "C": [], "s": [], "df": []}
        }
        dict_predictive = {
            "t": [], "prior": [], "y": [], "f": [], "q": [], "e": [], "df": [],
            "H": [], "L": [], "l": [], "what_detected": []}
        lt_model_history = []

        # Get the Bayes Factor function
        bf = self._bayes_factor(distr_fam=distr_fam, distr_type=distr_type)

        # Fitting the model with the monitor off
        for t in range(0, self._prior_length):
            # Keep model history
            lt_model_history.append(copy.deepcopy(mod))

            # The observation at time t
            yt = pd_y.values[t]
            Xt = pd_X.values[t, :]

            # Get mean and variance one step ahead of forecast distribution
            ft, qt = mod.forecast_marginal(k=1, X=Xt, state_mean_var=True)
            ft = ft[0][0]
            qt = qt[0][0]
            et = (yt - ft) / np.sqrt(qt)

            # Saving prior state parameters
            dict_state_parms["prior"]["a"].append(mod.a)
            dict_state_parms["prior"]["R"].append(mod.R)

            # Update model
            mod.update(y=yt, X=Xt)

            # Saving posterior state parameters
            dict_state_parms["posterior"]["m"].append(mod.m)
            dict_state_parms["posterior"]["C"].append(mod.C)
            dict_state_parms["posterior"]["s"].append(mod.s[0][0])
            dict_state_parms["posterior"]["df"].append(mod.n-1)

            # Saving predictive parameters
            dict_predictive["prior"].append(True)
            dict_predictive["t"].append(t+1)
            dict_predictive["y"].append(yt)
            dict_predictive["f"].append(ft)
            dict_predictive["q"].append(qt)
            dict_predictive["e"].append(et)
            dict_predictive["df"].append(mod.n-1)
            dict_predictive["H"].append(1)
            dict_predictive["L"].append(1)
            dict_predictive["l"].append(1)
            dict_predictive["what_detected"].append("nothing")

        # Fitting the model with the monitor on
        for t in range(self._prior_length, n):
            # Keep model history
            lt_model_history.append(copy.deepcopy(mod))

            # The observation at time t
            yt = pd_y.values[t]
            Xt = pd_X.values[t, :]

            # Get mean and variance one step ahead of forecast distribution
            ft, qt = mod.forecast_marginal(k=1, X=Xt, state_mean_var=True)
            ft = ft[0][0]
            qt = qt[0][0]

            # Standardize error
            et = (yt - ft) / np.sqrt(qt)

            # Bayes factor (Ht)
            Ht = bf(e=et, delta=h, df=mod.n-1)

            # Cumulative Bayes factor (Lt)
            Lt = 1
            if t > 0:
                Lt = Ht * np.min([1, dict_predictive["L"][t - 1]])

            # Run length (lt)
            lt = 1
            if t > 0:
                lt = np.where(dict_predictive["L"][t - 1] < 1,
                              dict_predictive["l"][t - 1] + 1, 1)

            dict_predictive["what_detected"].append("nothing")
            # Check structural change in parameters
            if (Ht >= tau) & ((Lt < tau) | (lt > 2)):
                dict_predictive["what_detected"][t] = "parametric_change"
                if self._verbose:
                    msg = ("Parametric change detected at time {t} with "
                           "H={Ht:.4e}, L={Lt:.4e} and l={lt}")
                    print(msg.format(t=t+1, Ht=Ht, lt=lt, Lt=Lt))
                # Change prior variance and get back in time (t - lt + 1)
                index = t - lt + 1
                mod = copy.deepcopy(lt_model_history[index])
                self._increase_uncertainty(
                    model=mod, discount_factors=self._discount_intervention)
                for i in range(index, t):
                    ft, qt = mod.forecast_marginal(k=1, X=pd_X.values[i, :],
                                                   state_mean_var=True)
                    dict_predictive["f"][i] = ft[0][0]
                    dict_predictive["q"][i] = qt[0][0]
                    mod.update(y=pd_y.values[i], X=pd_X.values[i, :])

                Lt = 1
                lt = 0
                # Compute new adjust forecast mean and variance
                ft, qt = mod.forecast_marginal(k=1, X=Xt, state_mean_var=True)
                ft = ft[0][0]
                qt = qt[0][0]

            # Check potential outlier
            potential_outlier = False
            if ((Lt < tau) & (lt == 1)):
                dict_predictive["what_detected"][t] = "outlier"
                if self._verbose:
                    msg = ("Potential outlier detected at time {t} with "
                           "H={Ht:.4e}, L={Lt:.4e} and l={lt}")
                    print(msg.format(t=t+1, Ht=Ht, lt=lt, Lt=Lt))
                yt = None
                Lt = 1
                lt = 0
                potential_outlier = True

            # Saving prior state parameters
            dict_state_parms["prior"]["a"].append(mod.a)
            dict_state_parms["prior"]["R"].append(mod.R)

            # Update model
            mod.update(y=yt, X=Xt)

            # Increase uncertainty
            if potential_outlier:
                self._increase_uncertainty(
                    model=mod, discount_factors=self._discount_intervention)

            # Saving posterior state parameters
            dict_state_parms["posterior"]["m"].append(mod.m)
            dict_state_parms["posterior"]["C"].append(mod.C)
            dict_state_parms["posterior"]["s"].append(mod.s[0][0])
            dict_state_parms["posterior"]["df"].append(mod.n-1)

            # Saving predictive parameters
            dict_predictive["t"].append(t+1)
            dict_predictive["prior"].append(False)
            dict_predictive["y"].append(pd_y.values[t])
            dict_predictive["f"].append(ft)
            dict_predictive["q"].append(qt)
            dict_predictive["e"].append(et)
            dict_predictive["df"].append(mod.n-1)
            dict_predictive["H"].append(Ht)
            dict_predictive["L"].append(Lt)
            dict_predictive["l"].append(lt)
            # end loop

        df_predictive = pd.DataFrame(dict_predictive)

        # Organize the posterior parameters
        df_posterior = tidy_parameters(
            dict_parameters=dict_state_parms["posterior"],
            entry_m="m", entry_v="C",
            names_parameters=list(mod.get_coef().index),
            index_seas_parameters=mod.iseas,
            F=mod.F)
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

        # Compute credible intervals
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

        # Output
        out = {"predictive": df_predictive, "posterior": df_posterior}

        # Compute smooth distribution
        if self._smooth:
            smooth_class = Smoothing(
                mod=self._mod, interval=self._interval, level=self._level)
            dict_smooth = smooth_class.fit(
                y=pd_y, dict_state_parms=dict_state_parms)
            out = {"filter": out, "smooth": dict_smooth}

        mod.deltrend = self._mod.deltrend
        mod.delregn = self._mod.delregn
        mod.delseas = self._mod.delseas
        out["model"] = mod

        return out

    def _fit_bilateral(self, h, tau, distr_fam, distr_type):
        mod = copy.deepcopy(self._mod)
        pd_y = self._pd_y.copy()
        pd_X = self._pd_X.copy()
        n = len(pd_y)

        # Dictionaries to keep state and predictive parameters
        dict_state_parms = {
            "prior": {"a": [], "R": []},
            "posterior": {"m": [], "C": [], "s": [], "df": []}
        }
        dict_predictive = {
            "t": [], "prior": [], "y": [], "f": [], "q": [], "e": [], "df": [],
            "H_lower": [], "L_lower": [], "l_lower": [],
            "H_upper": [], "L_upper": [], "l_upper": [],
            "what_detected": []}
        lt_model_history = []

        # Get the Bayes Factor function
        bf = self._bayes_factor(distr_fam=distr_fam, distr_type=distr_type)

        # Fitting the model with the monitor off
        for t in range(0, self._prior_length):
            # Keep model history
            lt_model_history.append(copy.deepcopy(mod))

            # The observation at time t
            yt = pd_y.values[t]
            Xt = pd_X.values[t, :]

            # Get mean and variance one step ahead of forecast distribution
            ft, qt = mod.forecast_marginal(k=1, X=Xt, state_mean_var=True)
            ft = ft[0][0]
            qt = qt[0][0]
            et = (yt - ft) / np.sqrt(qt)

            # Saving prior state parameters
            dict_state_parms["prior"]["a"].append(mod.a)
            dict_state_parms["prior"]["R"].append(mod.R)

            # Update model
            mod.update(y=yt, X=Xt)

            # Saving posterior state parameters
            dict_state_parms["posterior"]["m"].append(mod.m)
            dict_state_parms["posterior"]["C"].append(mod.C)
            dict_state_parms["posterior"]["s"].append(mod.s[0][0])
            dict_state_parms["posterior"]["df"].append(mod.n-1)

            # Saving predictive parameters
            dict_predictive["t"].append(t+1)
            dict_predictive["prior"].append(True)
            dict_predictive["y"].append(yt)
            dict_predictive["f"].append(ft)
            dict_predictive["q"].append(qt)
            dict_predictive["e"].append(et)
            dict_predictive["df"].append(mod.n-1)
            dict_predictive["H_upper"].append(1)
            dict_predictive["H_lower"].append(1)
            dict_predictive["L_upper"].append(1)
            dict_predictive["L_lower"].append(1)
            dict_predictive["l_upper"].append(1)
            dict_predictive["l_lower"].append(1)
            dict_predictive["what_detected"].append("nothing")

        for t in range(self._prior_length, n):
            # Keep model history
            lt_model_history.append(copy.deepcopy(mod))

            # The observation at time t
            yt = pd_y.values[t]
            Xt = pd_X.values[t, :]

            # Get mean and variance one step ahead of forecast distribution
            ft, qt = mod.forecast_marginal(k=1, X=Xt, state_mean_var=True)
            ft = ft[0][0]
            qt = qt[0][0]

            # Standardize error
            et = (yt - ft) / np.sqrt(qt)

            # Bayes factor (Ht): Upper = 0 and Lower = 1
            Ht = [bf(e=et, delta=h, df=mod.n-1),
                  bf(e=et, delta=-h, df=mod.n-1)]

            # Cumulative Bayes factor (Lt)
            Lt = [1, 1]
            if t > 0:
                Lt[0] = Ht[0] * np.min([1, dict_predictive["L_upper"][t - 1]])
                Lt[1] = Ht[1] * np.min([1, dict_predictive["L_lower"][t - 1]])

            # Run length (lt)
            lt = [1, 1]
            if t > 0:
                lt[0] = np.where(dict_predictive["L_upper"][t - 1] < 1,
                                 dict_predictive["l_upper"][t - 1] + 1, 1)
                lt[1] = np.where(dict_predictive["L_lower"][t - 1] < 1,
                                 dict_predictive["l_lower"][t - 1] + 1, 1)

            min_Ht = min(Ht)
            min_Lt = min(Lt)
            max_lt = max(lt)

            dict_predictive["what_detected"].append("nothing")
            # Check structural change in parameters
            if (min_Ht >= tau) & ((min_Lt < tau) | (max_lt > 2)):
                arg = np.argmin(Lt)
                side = "Upper" if arg == 0 else "Lower"
                dict_predictive["what_detected"][t] = (
                    side + "_parametric_change")
                if self._verbose:
                    msg = (side + " parametric change detected at time {t} "
                           "with H={Ht:.4e}, L={Lt:.4e} and l={lt}")
                    print(msg.format(t=t+1, Ht=min_Ht, lt=max_lt, Lt=min_Lt))
                # Change prior variance and get back in time (t - lt + 1)
                index = t - max_lt + 1
                mod = copy.deepcopy(lt_model_history[index])
                self._increase_uncertainty(
                    model=mod, discount_factors=self._discount_intervention)
                for i in range(index, t):
                    ft, qt = mod.forecast_marginal(k=1, X=pd_X.values[i, :],
                                                   state_mean_var=True)
                    dict_predictive["f"][i] = ft[0][0]
                    dict_predictive["q"][i] = qt[0][0]
                    mod.update(y=pd_y.values[i], X=pd_X.values[i, :])
                Lt[arg] = 1
                lt[arg] = 0
                # Compute new adjust forecast mean and variance
                ft, qt = mod.forecast_marginal(k=1, X=Xt, state_mean_var=True)
                ft = ft[0][0]
                qt = qt[0][0]

            # Check potential outlier
            potential_outlier = False
            if ((min_Lt < tau) & (max_lt == 1)):
                arg = np.argmin(Lt)
                side = "Upper" if arg == 0 else "Lower"
                dict_predictive["what_detected"][t] = side + "_outlier"
                if self._verbose:
                    msg = (side + " potential outlier detected at time {t} "
                           "with H={Ht:.4e}, L={Lt:.4e} and l={lt}")
                    print(msg.format(t=t+1, Ht=min_Ht, lt=max_lt, Lt=min_Lt))
                yt = None
                Lt[arg] = 1
                lt[arg] = 0
                potential_outlier = True

            # Saving prior state parameters
            dict_state_parms["prior"]["a"].append(mod.a)
            dict_state_parms["prior"]["R"].append(mod.R)

            # Update model
            mod.update(y=yt, X=Xt)

            if potential_outlier:
                self._increase_uncertainty(
                    model=mod, discount_factors=self._discount_intervention)

            # Saving posterior state parameters
            dict_state_parms["posterior"]["m"].append(mod.m)
            dict_state_parms["posterior"]["C"].append(mod.C)
            dict_state_parms["posterior"]["s"].append(mod.s[0][0])
            dict_state_parms["posterior"]["df"].append(mod.n-1)

            # Saving predictive parameters
            dict_predictive["t"].append(t+1)
            dict_predictive["prior"].append(False)
            dict_predictive["y"].append(pd_y.values[t])
            dict_predictive["f"].append(ft)
            dict_predictive["q"].append(qt)
            dict_predictive["e"].append(et)
            dict_predictive["df"].append(mod.n-1)
            dict_predictive["H_upper"].append(Ht[0])
            dict_predictive["H_lower"].append(Ht[1])
            dict_predictive["L_upper"].append(Lt[0])
            dict_predictive["L_lower"].append(Lt[1])
            dict_predictive["l_upper"].append(lt[0])
            dict_predictive["l_lower"].append(lt[1])
            # end loop

        df_predictive = pd.DataFrame(dict_predictive)

        # Organize the results of posterior distribution
        df_posterior = tidy_parameters(
            dict_parameters=dict_state_parms["posterior"],
            entry_m="m", entry_v="C",
            names_parameters=list(mod.get_coef().index),
            index_seas_parameters=mod.iseas,
            F=mod.F)
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

        # Compute credible intervals
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

        # Output
        out = {"predictive": df_predictive, "posterior": df_posterior}

        # Compute smooth distribution
        if self._smooth:
            smooth_class = Smoothing(
                mod=self._mod, interval=self._interval, level=self._level)
            dict_smooth = smooth_class.fit(
                y=pd_y, dict_state_parms=dict_state_parms)
            out = {"filter": out, "smooth": dict_smooth}

        mod.deltrend = self._mod.deltrend
        mod.delregn = self._mod.delregn
        mod.delseas = self._mod.delseas
        out["model"] = mod

        return out

    def _increase_uncertainty(self, model: pybats.dglm.dlm,
                              discount_factors: dict) -> None:
        """Increase the uncertainty of R matrix using discount factor.

        Parameters
        ----------
        model : pybats.dglm.dlm
            An object of class `pybats.dglm.dlm` with the defined DLM.
        discount_factors : dict
            Dictionary with discount factors values for `trend`, `seasonal`,
            and `reg` block.

        Returns
        -------
        None
            This function just change the model.R and model.W matrices.
        """
        model.deltrend = discount_factors["trend"]
        model.delseas = discount_factors["seasonal"]
        model.delregn = discount_factors["reg"]
        mat_discount = model.build_discount_matrix()
        # Discount information for prior variance at time  t + 1
        model.W = model.R / mat_discount - model.R
        model.R = model.R + model.W

    def _bayes_factor(self, distr_fam, distr_type):
        switcher = {
            "normal":
                {"location": self._bf_normal_location,
                 "scale": self._bf_normal_scale},
            "tstudent":
                {"location": self._bf_tstudent_location,
                 "scale": self._bf_tstudent_scale}
        }
        return switcher[distr_fam][distr_type]

    def _bf_normal_location(self, e, delta, df):
        return np.exp(0.5 * (delta ** 2) - e * delta)

    def _bf_tstudent_location(self, e, delta, df):
        p0 = stats.t.pdf(e, df=df)
        p1 = stats.t.pdf(e, loc=delta, df=df)
        return p0 / p1

    def _bf_normal_scale(self, e, delta, df):
        return delta * np.exp(-0.5 * e**2 * (1 - 1/delta**2))

    def _bf_tstudent_scale(self, e, delta, df):
        p0 = stats.t.pdf(e, df=df)
        p1 = stats.t.pdf(e, scale=delta, df=df)
        return p0 / p1
