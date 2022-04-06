"""Functions to load literature data."""
import os
import pandas as pd


def load_cp6():
    """
    Load CP6 time series.

    Monthly total sales, in monetary terms on a standard scale, of tobacco
    and related products marketed by a major company in the UK.
    The time of the data runs from January 1955 to December 1959 inclusive.
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    tmp = pd.read_csv(data_dir + 'cp6__west_harrison.csv')
    tmp["time"] = pd.to_datetime(tmp["time"])
    return tmp


def load_telephone_calls():
    """
    Load telephone calls time series.

    Monthly average daily calls to Cincinnati directory assistance from
    January 1962 to December 1973.
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    tmp = pd.read_csv(data_dir + 'telephone_calls__pankratz.csv')
    tmp["time"] = pd.to_datetime(tmp["time"])
    return tmp


def load_air_passengers():
    """
    Load AirPassengers time series.

    Monthly totals of international airline passengers, 1949 to 1960.
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    tmp = pd.read_csv(data_dir + 'air_passengers.csv')
    tmp["time"] = pd.to_datetime(tmp["time"])
    return tmp


def load_market_share():
    """
    Load Market Share time series.

    Weekly market share for a consumer product, January 1990 to December 1991.
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/data/'
    tmp = pd.read_csv(data_dir + 'pole_market_share.csv')
    tmp['time'] = pd.to_datetime(tmp["time"])
    return tmp
