"""Utils functions."""
import numpy as np
import pandas as pd


def tidy_posterior_parms(dict_posterior_parms: dict, entry_m: str,
                         entry_v: str):
    """Transform the state space moments from dictionary of lists to\
    tidy pd.DataFrame in long format.

    Parameters
    ----------
    dict_posterior_parms : dict
        dictionary with the posterior (m and C) and prior (a and R) moments
        for the state space parameters along time.
    entry_m : str
        Entrie name in the `dict_posterior_parms` of mean vector of
        state space parameters.
    entry_v : str
        Entrie name in the `dict_posterior_parms` of covariance matrix of
        state space parameters.

    Returns
    -------
    pd.DataFrame
        A DataFrame in tidy format with columns parameter, mean, and
        variance for each time.

    """

    def _get_mean(x: np.ndarray):
        """Get the mean vector from np.nadarray and transform in DataFrame.

        Parameters
        ----------
        x : np.ndarray
            The mean vector of state space parameters. It could be the prior or
            posterior moments.

        Returns
        -------
        pd.DataFrame
            The extracted mean values.

        """
        index = ["theta_" + str(i+1) for i in range(0, x.shape[0])]
        return pd.DataFrame(data={"mean": x[:, 0]}, index=index)

    def _get_var(x: np.ndarray):
        """Get the diagonal elements of covariance matrix from np.nadarray\
        and transform in DataFrame.

        Parameters
        ----------
        x : np.ndarray
            The covariance matrix of state space parameters.
            It could be the prior or posterior moments.

        Returns
        -------
        pd.DataFrame
            The extracted variance values.

        """
        index = ["theta_" + str(i+1) for i in range(0, x.shape[0])]
        return pd.DataFrame(data={"variance": np.diag(x)}, index=index)

    df_mean_parms = pd.concat(
        list(map(_get_mean, dict_posterior_parms[entry_m])))
    df_var_parms = pd.concat(
        list(map(_get_var, dict_posterior_parms[entry_v])))
    df_posterior = pd.concat(
        [df_mean_parms.reset_index(),
         df_var_parms.reset_index(drop=True)], axis=1)
    df_posterior.rename(columns={"index": "parameter"}, inplace=True)

    return df_posterior[["parameter", "mean", "variance"]]
