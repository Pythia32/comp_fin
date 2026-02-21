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


#### Step 1:
## Load SPX (call) option quote dataset
fp = "data/" + "option20230201_20230228.csv" # File path
SPX_data = pd.read_csv(fp)
SPX_data = SPX_data[["date","exdate","symbol","strike_price","best_bid","best_offer","impl_volatility","delta","cp_flag"]] # Keep only relevant features


## Preprocessing
SPX_calls = SPX_data[SPX_data["cp_flag"] == "C"].copy() # Keep calls only

SPX_calls["date"] = pd.to_datetime(SPX_calls["date"]) # Consistent data type for temporal features
SPX_calls["exdate"] = pd.to_datetime(SPX_calls["exdate"])
SPX_calls["strike"] = SPX_calls["strike_price"] / 1000 # Rescale strikes to match index-point units
SPX_calls.drop(columns="strike_price", inplace=True)
SPX_calls["midquote"] = (SPX_calls["best_bid"] + SPX_calls["best_offer"]) / 2 # Construct midquote

SPX_calls_no_nan = SPX_calls.dropna(subset=["impl_volatility"]) # Remove observations with missing (NaN) or invalid (<=0) implied volatility
SPX_calls_no_nan = SPX_calls_no_nan[SPX_calls_no_nan["impl_volatility"] > 0]


## Reporting (1)
print("Raw observation count: ", SPX_data.shape[0])

print("Observation count after filtering to calls: ", SPX_calls.shape[0])
print("\nMidquote (V_t) missing / incorrect data rate: ",
  (
      SPX_calls["best_bid"].isna() | # Missing entries
      SPX_calls["best_offer"].isna() |
      (SPX_calls["best_bid"] < 0) | # Best bid may be 0 (no buyers), but should never be negative
      (SPX_calls["best_offer"] < 0) | # Best offer should always be positive, as the value of a European call option is always at least 0, and giving away such an option for free is not sensible
      (SPX_calls["best_bid"] >= SPX_calls["best_offer"]) # This should result in a market order, clearing the associated bid/offer
  ).mean())
print("Implied volatility (sigma_mkt) missing / incorrect data rate: ", (SPX_calls["impl_volatility"].isna() | (SPX_calls["impl_volatility"] <= 0)).mean())
print("Delta missing data rate: ", SPX_calls["delta"].isna().mean())

print("\nFinal observation count after preprocessing: ", SPX_calls_no_nan.shape[0])
print("\nUnique trading dates (after preprocessing): ", SPX_calls_no_nan["date"].nunique())
print("Unique expiries (after preprocessing): ", SPX_calls_no_nan["exdate"].nunique())
print("\nFinal columns: ", SPX_calls_no_nan.columns)


## Further Data Augmenting..
# spx_close = fetch_spx_close("2023-02-01", "2023-03-01")
# print(spx_close)



#### Step 2
contracts = SPX_calls_no_nan['symbol'].unique().tolist()
dates = SPX_calls_no_nan["date"].unique().tolist()

start_dates = dates[:-1:]
end_dates = dates[1::]
sequential_date_pairs = list(zip(start_dates, end_dates))

## Construct DataFrame containing the midquote time series for each contract (row) and across all dates (column) in the preprocessed dataset.
SPX_call_midquote_series = (  # DataFrame of V_t, where each row corresponds to a unique contract ("symbol"), and each column corresponds to a unique date ("date")
    SPX_calls_no_nan
    .pivot(index="symbol", columns="date", values="midquote")
    .reindex(index=contracts)
)

SPX_call_midquote_series.columns.name = "date"
SPX_call_midquote_series.index.name = "contract (symbol)"

SPX_call_midquote_series_no_nan = SPX_call_midquote_series.dropna() # Remove observations (contracts) with incomplete time series (having NaN entries)

## Construct DataFrame containing the midquote delta (sequential difference) time series for each contract (row) and across all dates (column) in the preprocessed dataset.
delta_series_data = []
for (start, end) in sequential_date_pairs:
    deltas_at_time_t = SPX_call_midquote_series[end] - SPX_call_midquote_series[start]
    delta_series_data.append(deltas_at_time_t)
SPX_call_delta_series = pd.concat(delta_series_data, axis=1)  # DataFrame of delta V_t = V_{t+1} - V_t, where each row corresponds to a unique contract ("symbol"), and each column corresponds to a unique pair (start_date, end_date), specifying the location of the time-increment where the difference (delta) was computed.

SPX_call_delta_series.columns = start_dates # Rename columns to match the corresponding dates
SPX_call_delta_series.columns.name = "date"
SPX_call_delta_series.index.name = "contract (symbol)"

SPX_call_delta_series_no_nan = SPX_call_delta_series.dropna() # Remove observations (contracts) with incomplete time series (having NaN entries)

# Consistency check: number of observations (rows) in midquote time series df and in midquote delta time series df should be equal
if (SPX_call_midquote_series_no_nan.shape[0] != SPX_call_delta_series_no_nan.shape[0]):
    raise ValueError(
        f"\nConsistency check failed: "
        f"SPX_call_midquote_series_no_nan has {SPX_call_midquote_series_no_nan.shape[0]} rows,"
        f"SPX_call_delta_series_no_nan has {SPX_call_delta_series_no_nan.shape[0]} rows. These should be equal."
    )


## Reporting (2)
# Summary statistics of V_t and delta V_t

# Number of contracts retained(?) after constructing delta V_t
print("\nNumber of unique contracts in the preprocessed dataset: ", SPX_call_delta_series.shape[0])
print("Number of contracts retained (complete time series) after constructing delta V_t: ", SPX_call_delta_series_no_nan.shape[0]) # Note: retained contracts have a complete time series, discarded contracts do not



#### Step 3:
## SPX close series
start = "2023-02-01"
end = "2023-03-02"

d1 = pd.Timestamp(start).strftime("%Y%m%d")
d2 = pd.Timestamp(end).strftime("%Y%m%d")

url = f"https://stooq.com/q/d/l/?s=^spx&d1={d1}&d2={d2}&i=d"
df = pd.read_csv(url, parse_dates=["Date"]).sort_values("Date")

SPX_price_series = df.set_index("Date")["Close"]
print(SPX_price_series)
# print(type(SPX_price_series))

# print(SPX_call_midquote_series_no_nan)


## Delta SPX close series
next = SPX_price_series.shift(-1)
# print(next)
delta_SPX_price_series = (next - SPX_price_series).dropna()
print(delta_SPX_price_series)


## Reporting
# Merge coverage with (V_t/ delta V_t)
# ... TO DO!

# Summary statistics
print(SPX_price_series.describe())
print(delta_SPX_price_series.describe())

# Plots of S_t and delta S_t over time
SPX_price_series.plot(title="SPX Close Price")
# plt.show()

delta_SPX_price_series.plot(title="Daily SPX Change")
# plt.show()

print(dates)
print(SPX_call_midquote_series_no_nan)
print(SPX_call_delta_series_no_nan)
print(type(SPX_call_midquote_series_no_nan))
