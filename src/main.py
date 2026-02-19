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


#### Step 2:
## Construct Contract Time Series
contracts = calls['symbol'].unique().tolist()
# print(type(contracts))
# print(contracts)

dates = calls["date"].unique().tolist()
# print(type(dates[0]))
# print(dates)

dates_start = dates[:-1:]
dates_end = dates[1::]
date_delta_pairs = list(zip(dates_start, dates_end))

print("Date missing data rate: ", calls["date"].isna().mean())  # No dates missing! :)

# print(dates)
# print(dates[:-1:])

# c = contracts[0]
# d = dates[0]
# test = calls[
#       (calls["symbol"]==c) & 
#       (calls["date"]==d)
#       ]["midquote"] #.item()
# print(test)
# print(type(test))
# print(test.size)

### Step 2/ attempt 4:  SUCCESS + FAST!!
contract_ts_df = (  # DataFrame of V_t, where each row corresponds to a unique contract ("symbol"), and each column corresponds to a unique date ("date")
    calls
    .pivot(index="symbol", columns="date", values="midquote")
    .reindex(index=contracts)
)
contract_delta_ts = []
for (start, end) in date_delta_pairs:
    delta_v_vec = contract_ts_df[end] - contract_ts_df[start]
    contract_delta_ts.append(delta_v_vec)

contract_delta_ts_df = pd.concat(contract_delta_ts, axis=1)  # DataFrame of delta V_t = V_{t+1} - V_t, where each row corresponds to a unique contract ("symbol"), and each column corresponds to a unique pair (start_date, end_date), specifying the location of the time-increment where the difference (delta) was computed.
print(contract_delta_ts_df)
# print(contract_ts_df)


## Reporting
# Summary statistics of V_t and delta V_t

# Number of contracts retained(?) after constructing delta V_t
contract_delta_ts_df_no_nan = contract_delta_ts_df.dropna()
print("/nNumber of contracts after preprocessing: ", contract_delta_ts_df.shape[0])
print("Number of contracts retained after constructing delta V_t: ", contract_delta_ts_df_no_nan.shape[0])

### Check consistency for number of rows containing NaN's between V_t and delta V_t
contract_ts_df_no_nan = contract_ts_df.dropna()
print(contract_ts_df_no_nan.shape[0])  # == 4975, hence the number of rows containing NaN entries in V_t is consistent with that in delta V_t.

### --> Question: How many contracts do not exist before some date d; but have otherwise complete entries for dates' remainder?
columns = list(contract_delta_ts_df.columns)
# print(columns)
for i in columns:
    test_df = contract_delta_ts_df.iloc[:, i:].dropna()
    print(f"Number of rows retained including columns starting from {i}: {test_df.shape[0]}") # Number of rows retained is strictly increasing as i increases. Ultimately we would retain a total 7841 rows if we were to retain also entries for which at least one delta V_t entry is not NaN and all subsequent entries (increasing columns) are also not NaN.

## // Legacy Code for Step 2: //


### Step 2/ attempt 3:  SUCCESS (But SLOW)
# calls_indexed = calls.set_index(["symbol", "date"])["midquote"]
# contract_ts_list = []
# for c in contracts:
#   row = []
#   for (start, end) in date_delta_pairs:
#       v_start = calls_indexed.get((c, start))
#       v_end   = calls_indexed.get((c, end))
#       if v_start is None or v_end is None:
#           row.append(np.nan)
#       elif isinstance(v_start, float) and isinstance(v_end, float):
#           delta_v = v_end - v_start
#           row.append(delta_v)
#           # print(v_start.size) ###
#           # print(v_start.item()) ###
#       else:
#         raise TypeError("Local variables v_start and v_end should be floats (or boolean with value None)")
#   contract_ts_list.append(row)
# print(contract_ts_list)

### attempt 3/ Testing:
# c = contracts[100]
# print(c)
# test_data = calls[calls["symbol"]==c]["date"]
# print(test_data)
# # print(dates[11])
# d_early = dates[10]
# d_late = dates[11]
# test1 = calls_indexed.get((c,d_early))
# print(test1)  # Should(!) return a float
# test2 = calls_indexed.get((c,d_late))
# print(test2) # Returns None (because searched entry index does not exist)
# #//
# (start, end) = date_delta_pairs[10]
# print(start, end)

### Step 2/ attempt 2:  FAIL
# contract_ts_list = []
# for c in contracts:
#   row = []
#   for (start, end) in date_delta_pairs:
#     try:
#       delta_V_t = calls[
#         (calls["symbol"]==c) & 
#         (calls["date"]==end)
#         ]["midquote"].item() - calls[
#         (calls["symbol"]==c) & 
#         (calls["date"]==start)
#         ]["midquote"].item()
#       print(delta_V_t) ###
#       row.append(delta_V_t)
#     except ValueError as e:
#       if "can only convert an array of size 1 to a Python scalar" in str(e):
#         print("Missing entry detected, substituting NaN entry")
#         row.append(np.nan)
#       else:
#         raise # Raise if unexpected ValueError
#     except Exception as e: # Catches and identifies any potential exceptions not yet observed.
#       print("Exception was: ", e)
#       print(type(e))
#   contract_ts_list.append(row)
# print(contract_ts_list)

### Step 2/ attempt 1:  FAIL
# contract_ts_list = []
# for c in contracts:
#   row = []
#   for (start, end) in date_delta_pairs:
#     delta_V_t = calls[
#       (calls["symbol"]==c) & 
#       (calls["date"]==end)
#       ]["midquote"] - calls[
#       (calls["symbol"]==c) & 
#       (calls["date"]==start)
#       ]["midquote"]
#     if delta_V_t.size==1:
#       print(delta_V_t.item())
#       row.append(delta_V_t.item())
#     elif delta_V_t.size==0:
#       print("Missing entry")
#       row.append(np.nan)
#     elif delta_V_t.size>=2:
#       # print("ERROR: too many entries")
#       raise ValueError(f"Multiple entries found for symbol={c}, date={d}")
#     else:
#       raise ValueError(f"Incompatible data found for symbol={c}, date={d}")
#   contract_ts_list.append(row)


# contract_ts_df = pd.DataFrame(contract_ts_list)
