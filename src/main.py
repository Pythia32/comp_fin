import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
from skfolio.datasets import load_sp500_dataset, load_sp500_implied_vol_dataset
from skfolio.preprocessing import prices_to_returns
import datetime
import math
from yahooquery import Ticker

from utils import fetch_spx_close, fetch_vix_close


## Data Loading
data_path = "data/" + "option20230201_20230228.csv"
data_SPX = pd.read_csv(data_path) # Load SPX option quote data
data_SPX = data_SPX[["date","exdate","symbol","strike_price","best_bid","best_offer","impl_volatility","delta","cp_flag"]] # Discard irrelevant features
print("Raw data observation count: ", data_SPX.shape[0]) # Reporting..


## Data Preprocessing
calls = data_SPX[data_SPX["cp_flag"] == "C"].copy() # Keep calls only
print("Observation count after filtering to calls: ", calls.shape[0]) # Reporting..

calls["date"] = pd.to_datetime(calls["date"]) # Convert entry type for "date features" to datetime64[ns] for consistency
calls["exdate"] = pd.to_datetime(calls["exdate"])
calls["strike"] = calls["strike_price"] / 1000 # Rescale strikes to match index-point units, i.e. division by 1000
calls.drop(columns="strike_price", inplace=True)
calls["midquote"] = (calls["best_bid"] + calls["best_offer"]) / 2 # Construct midquote
# print("\nMidquote (V_t) missing rate: ", calls["midquote"].isna().mean())
print("\nMidquote (V_t) missing / incorrect data rate: ",
  (
      calls["best_bid"].isna() | # Missing entries
      calls["best_offer"].isna() |
      (calls["best_bid"] < 0) | # Best bid may be 0 (no buyers), but should never be negative
      (calls["best_offer"] < 0) | # Best offer should always be positive, as the value of a European call option is always at least 0, and giving away such an option for free is not sensible
      (calls["best_bid"] >= calls["best_offer"]) # This should result in a market order, clearing the associated bid/offer
  ).mean())
print("Implied volatility (sigma_mkt) missing / incorrect data rate: ", (calls["impl_volatility"].isna() | (calls["impl_volatility"] <= 0)).mean())
print("Delta missing data rate: ", calls["delta"].isna().mean())

calls = calls.dropna(subset=["impl_volatility"]) # Remove observations with missing (NaN) or invalid (<=0) implied volatility
calls = calls[calls["impl_volatility"] > 0]
print("\nFinal observation count after preprocessing: ", calls.shape[0]) # Reporting..
print("\nUnique trading dates (after preprocessing): ", calls["date"].nunique())
print("Unique expiries (after preprocessing): ", calls["exdate"].nunique())

print("\nFinal columns: ", calls.columns)


## Further Data Augmenting..
# spx_close = fetch_spx_close("2023-02-01", "2023-03-01")