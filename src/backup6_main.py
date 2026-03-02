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



#### Step 2:
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

SPX_call_midquote_series.columns.name = "Date"
SPX_call_midquote_series.index.name = "Contract (symbol)"

SPX_call_midquote_series_no_nan = SPX_call_midquote_series.dropna() # Remove observations (contracts) with incomplete time series (having NaN entries)

## Construct DataFrame containing the midquote delta (sequential difference) time series for each contract (row) and across all dates (column) in the preprocessed dataset.
delta_series_data = []
for (start, end) in sequential_date_pairs:
    deltas_at_time_t = SPX_call_midquote_series[end] - SPX_call_midquote_series[start]
    delta_series_data.append(deltas_at_time_t)
SPX_call_delta_series = pd.concat(delta_series_data, axis=1)  # DataFrame of delta V_t = V_{t+1} - V_t, where each row corresponds to a unique contract ("symbol"), and each column corresponds to a unique pair (start_date, end_date), specifying the location of the time-increment where the difference (delta) was computed.

SPX_call_delta_series.columns = start_dates # Rename columns to match the corresponding dates
SPX_call_delta_series.columns.name = "Date"
SPX_call_delta_series.index.name = "Contract (symbol)"

SPX_call_delta_series_no_nan = SPX_call_delta_series.dropna() # Remove observations (contracts) with incomplete time series (having NaN entries)

# Consistency check: number of observations (rows) in midquote time series df and in midquote delta time series df should be equal
assert SPX_call_midquote_series_no_nan.shape[0] == SPX_call_delta_series_no_nan.shape[0], "Number of rows in V_t and delta V_t (no NaN) DataFrames should be equal"
# if (SPX_call_midquote_series_no_nan.shape[0] != SPX_call_delta_series_no_nan.shape[0]):
#     raise ValueError(
#         f"\nConsistency check failed: "
#         f"SPX_call_midquote_series_no_nan has {SPX_call_midquote_series_no_nan.shape[0]} rows,"
#         f"SPX_call_delta_series_no_nan has {SPX_call_delta_series_no_nan.shape[0]} rows. These should be equal."
#     )


## Reporting (2)
# Summary statistics of V_t and delta V_t
# ...

# Number of contracts retained(?) after constructing delta V_t
print("\nNumber of unique contracts in the preprocessed dataset: ", SPX_call_delta_series.shape[0])
print("Number of contracts retained (complete time series) after constructing delta V_t: ", SPX_call_delta_series_no_nan.shape[0]) # Note: retained contracts have a complete time series, discarded contracts do not



#### Step 3:
## Fetch SPX close data across the relevant time frame, i.e. spanning dates contained in the preprocessed dataset
start = "2023-02-01"
end = "2023-02-28" # The preprocessed dataset contains no observations for dates beyond this point.

d1 = pd.Timestamp(start).strftime("%Y%m%d")
d2 = pd.Timestamp(end).strftime("%Y%m%d")

url = f"https://stooq.com/q/d/l/?s=^spx&d1={d1}&d2={d2}&i=d"
df = pd.read_csv(url, parse_dates=["Date"]).sort_values("Date")

## Construct time series of SPX close (price)
SPX_close_series = df.set_index("Date")["Close"]

## Construct time series of SPX close delta
next_close = SPX_close_series.shift(-1)
SPX_delta_series = (next_close - SPX_close_series).dropna()


## Merge datasets
Augmented_SPX_call_midquote_series = SPX_call_midquote_series_no_nan.copy()
Augmented_SPX_call_midquote_series.loc["SPX Close"] = SPX_close_series

Augmented_SPX_call_delta_series = SPX_call_delta_series_no_nan.copy()
Augmented_SPX_call_delta_series.loc["SPX Delta"] = SPX_delta_series


## Reporting (3)
# Merge coverage
print("\nSPX close/delta merge coverage: 1.0")

# Summary statistics
# print(SPX_close_series.describe())
# print(SPX_delta_series.describe())

# Plots of S_t and delta S_t over time
SPX_close_series.plot(title="SPX Close Price")
# plt.show()  # Can be improved! + UNCOMMENT THIS in the hand-in doc.

SPX_delta_series.plot(title="Daily SPX Change")
# plt.show()  # Can be improved! + UNCOMMENT THIS in the hand-in doc.



#### Step 4:
## Fetch the daily US treasury par yield curve rates
url = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rate-archives/par-yield-curve-rates-2020-2023.csv"
)
treasury = pd.read_csv(url)

## Preprocessing
treasury["date"] = pd.to_datetime(treasury["date"]).dt.normalize()

treasury = treasury.rename(columns={"date": "Date"})
treasury.set_index("Date", inplace=True)
treasury = treasury.loc[dates]

standardized_column_names = list(
    map(lambda x: 
        x.split()[0] + "/12" if x.split()[-1] == "mo"
        else x.split()[0],
        treasury.columns
    )
)
treasury.columns = standardized_column_names # Rename columns to match standardized format: fraction/number of years
treasury.columns.name = "Tenor (years)"

treasury /= 100 # Raw par yield curve rates are given as percentages, we convert to decimal
# treasury.iloc[:,1:] /= 100 # (Old)ERROR: the former "Date" column had already been set as index for treasury, hence excluding the first column was unnecessary (and in fact counterproductive!)

## Reindex SPX call (delta) dataframes by their contract characterization: (K,T) where K is the strike and T the expiry date
df = SPX_calls_no_nan[SPX_calls_no_nan["symbol"].isin(contracts)][["symbol","strike","exdate"]]
df = df.drop_duplicates()

group_sizes = df.groupby("symbol").size()
assert (group_sizes == 1).all(), \
    f"Multiple characterizations (K,T) detected for contracts (\"symbol\"): {group_sizes[group_sizes > 1].index.tolist()}"

contract_characterization = dict(zip( # Dictionary has format {contract ("symbol"): (K,T)}, where K is the strike and T the expiry date of the contract
    df["symbol"],
    zip(df["strike"], df["exdate"])
))

new_indexes = {key: (key, contract_characterization[key][0], contract_characterization[key][1]) for key in contract_characterization.keys()}

Augmented_SPX_call_midquote_series.rename(index=new_indexes, inplace=True)
Augmented_SPX_call_midquote_series.index.name = "(Contract (\"symbol\"), K, T)"

Augmented_SPX_call_delta_series.rename(index=new_indexes, inplace=True)
Augmented_SPX_call_delta_series.index.name = "(Contract (\"symbol\"), K, T)"

## Construct a new DataFrame containing the year-fraction maturity (tau) for each contract at every timestamp (date)
expiry = Augmented_SPX_call_midquote_series.index[:-1]
call_dates = Augmented_SPX_call_midquote_series.columns

tau_matrix = [
    [round((expiry[i][-1] - call_dates[j]).days / 365, 5) # Can change this later depending on the desired precision of tau
     for j in range(Augmented_SPX_call_midquote_series.shape[1])]
    for i in range(Augmented_SPX_call_midquote_series.shape[0] - 1)
]
SPX_call_tau_series = pd.DataFrame(
    tau_matrix,
    index=Augmented_SPX_call_midquote_series.index[:-1],
    columns=Augmented_SPX_call_midquote_series.columns
) 


## (a) Treasury date alignment and missing-day handling
treasury_dates = list(treasury.index)
assert treasury_dates == dates, "Dates captured in treasury DataFrame do not match those in the augmented V_t / delta V_t DataFrames" # The dates can be expected to be equal by construction of the treasury DataFrame


##################################################################################### Check code thoroughly from this point!!
## (b) Nelson-Siegel-Svensson (NSS) curve
def nss_basis(tau, tau1, tau2): # Note: during the fitting process, each observation will have tau corresponding to some tenor in treasury.columns
    tau = np.maximum(tau, 1 / 365) # avoid tau=0
    # tau = np.maximum(tau, 1e-6)

    g1 = (1 - np.exp(-tau/tau1)) / (tau/tau1)
    g2 = g1 - np.exp(-tau/tau1)
    g3 = (1 - np.exp(-tau/tau2)) / (tau/tau2) - np.exp(-tau/tau2)

    return np.column_stack([np.ones_like(tau), g1, g2, g3])


## (c) Daily calibration via grid search + conditional OLS
def parse_tenor(x):
    if "/" in x:
        a, b = x.split("/")
        return float(a)/float(b)
    return float(x)

tenors = np.array([parse_tenor(c) for c in treasury.columns])


def conditional_ols(y, tenors, tau1, tau2):
    X = nss_basis(tenors, tau1, tau2)
    beta, *_ = np.linalg.lstsq(X, y) # Is a np.ndarray
    residuals = y - X @ beta
    sse = residuals @ residuals
    return beta, sse

# (!) Might be able to improve the grids below: (!)
# tau1_grid = np.geomspace(1e-3, 1e-1, 1000) # This one is computationally too heavy to be ran while still editing the code, but may give better parameter estimations in the end (unless it overfits?) 
# tau2_grid = np.geomspace(1e-1, 1, 500)
# 
# tau1_grid = np.geomspace(1e-3, 1e-1, 200) # np.geomspace generates a grid with numbers spaced evenly on a logscale, which is convenient/ fitting in this case
# tau2_grid = np.geomspace(1e-1, 1, 100)
# 
tau1_grid = np.geomspace(0.05, 5, 100)
tau2_grid = np.geomspace(0.25, 25, 100)
# 
# tau1_grid = np.geomspace(1e-4, 10, 100)
# tau2_grid = np.geomspace(1e-4, 10, 100)
# 
# tau1_grid = np.linspace(0.1, 5, 30) # These grids for tau1 and tau2 were arbitrarily picked --->! Should improve this/ provide clarification!
# tau2_grid = np.linspace(0.1, 10, 40)
# ----> Note(!): We could consider using a collection of grids, i.e. different tau1/tau2_grid for every date ----> SideNote(!): This may lead to slight overfitting?/ violation of market assumptions?

def fit_nss_grid(y, tenors, tau1_grid, tau2_grid):

    best_sse = np.inf # np.inf is a representation of infinity in Python, hence a convenient starting value
    best_params = None

    for tau1 in tau1_grid:
        for tau2 in tau2_grid:
            if tau2 < tau1: # Only consider pairs (tau1, tau2) satisfying tau2 >= tau1, to ensure numerical stability
                continue

            beta, sse = conditional_ols(y, tenors, tau1, tau2)

            if sse < best_sse:
                best_sse = sse
                best_params = (beta, tau1, tau2)

    beta, tau1, tau2 = best_params
    return beta, tau1, tau2


nss_results = {} # For each date (key); contains the estimated parameters of the NSS curve (value)

for date in treasury.index:
    y = treasury.loc[date].values # .values method for pd.Series returns an ndarray object
    beta, tau1, tau2 = fit_nss_grid(y, tenors, tau1_grid, tau2_grid)

    nss_results[date] = np.concatenate([beta, [tau1, tau2]])

nss_df = pd.DataFrame(
    nss_results,
    index=["beta0","beta1","beta2","beta3","tau1","tau2"]
).T


## (d) Evaluating the curve at option maturities    ##### CHECK this one EVEN MORE CAREFULLY (!!!) (I was tired..)
def fitted_yield(tau, params):
    beta = params[:4]
    assert len(beta) == 4, "beta should have length 4"
    tau1 = params[4]
    tau2 = params[5]

    X = nss_basis(tau, tau1, tau2)
    # X = nss_basis(np.array([tau]), tau1, tau2)
    return X @ beta
    # return float(X @ beta)

# def fitted_yield(tau, beta, tau1, tau2):
#     X = nss_basis(np.array([tau]), tau1, tau2)
#     return float(X @ beta)

# def evaluate_curve(tau_matrix, beta, tau1, tau2):
#     tau_flat = tau_matrix.flatten() # To make it compatible (1d ndarray) with function nss_basis()
#     X = nss_basis(tau_flat, tau1, tau2)
#     fitted = X @ beta
#     return fitted.reshape(tau_matrix.shape) # "Deflatten" the result to match the original matrix shape

tau_matrix = np.array(tau_matrix, dtype=float) ### NOT necessarily REQUIRED, can be removed later

yields = {}
for date in SPX_call_tau_series.columns:
    tau = SPX_call_tau_series[date].values
    params = nss_df.loc[date].values
    row = fitted_yield(tau, params)
    yields[date] = row

SPX_call_fitted_yields = pd.DataFrame(
    yields,
    index = SPX_call_tau_series.index,
    columns = SPX_call_tau_series.columns
)

### Testing
# print(treasury)
# print(SPX_call_tau_series)
# print(tau_matrix)
# print(nss_df)
# print(SPX_call_fitted_yields)
###


## (e) Converting Treasury yield quotes to a continuously-compounded zero rate
def par_to_cc(y):
    return 2 * np.log(1 + y/2)

cc_rates = SPX_call_fitted_yields.map(par_to_cc)

## The associated discount factors
disc_factors = np.exp(-cc_rates * tau_matrix) # Element-wise multiplication


## Reporting (4)
# (4.1) Summary of {r_t(\tau)} values
print("\n", cc_rates.describe()) # Statistics per date
print(pd.Series(cc_rates.to_numpy().flatten()).describe()) # Statistics over all values

# (4.2) Plot of \tau\to r_t(\tau), for a representative date
repr_date = "2023-02-14" # This date is more in the "middle" w.r.t. the captured dates, perhaps a more suitable choice
# repr_date = "2023-02-28"
repr_date = pd.to_datetime(repr_date)

tau_series_no_dups = SPX_call_tau_series.drop_duplicates(subset=[repr_date])
cc_rates_no_dups = cc_rates.loc[tau_series_no_dups.index]

x = tau_series_no_dups[repr_date].values
y = cc_rates_no_dups[repr_date].values
x_sorted = np.sort(x)
sorted_idx = np.argsort(x)
y_sorted = y[sorted_idx]
plt.figure(figsize=(8, 5))
plt.scatter(x, y, label=r"Observed $r_t(\tau)$", zorder=3) # zorder=3 makes the dots from the scatterplots go "below" the line from the plot
plt.plot(x_sorted, y_sorted, label="Approximate NSS-fit")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$r_t(\tau)$", rotation="horizontal")
plt.title(label=r"Plot of observed $r_t(\tau)$ and the approximate NSS-fit " + f"on t={repr_date.date()}:")
plt.legend()
plt.grid(True)
# plt.show() # UNCOMMENT THIS in the hand-in doc.

# (4.3) Verification that every option row has a finite r_t(\tau) and discount factor P_t(\tau)
disc_factors_no_dups = disc_factors.drop_duplicates()

def val_checker(row: pd.Series): # Checks whether a DataFrame row (pd.Series) contains atleast one element that is both not NaN and not infinite (np.inf)
    flag = 0
    if row.isna().all() == 1 or np.isinf(row.values).all() == 1:
        flag = 1
    return flag

cc_val_check = cc_rates_no_dups.apply(val_checker, axis=0)
if any(cc_val_check):
    print("\nThere exists an option row having no finite r_t(\\tau)")
else:
    print("\nAll option rows have atleast one finite r_t(\\tau)")

disc_fac_val_check = disc_factors_no_dups.apply(val_checker, axis=0)
if any(disc_fac_val_check):
    print("There exists an option row having no finite P_t(\\tau)")
else:
    print("All option rows have atleast one finite P_t(\\tau)")



#### Step 5:
## (a) Baseline delta

# # Reconstruct a first DataFrame
# spx_call_delta_ts = Augmented_SPX_call_delta_series.iloc[:-1, :].copy()
# spx_call_delta_ts.columns.name = None
# idx, spx_call_delta_ts["strike"], spx_call_delta_ts["expiry"] = zip(*spx_call_delta_ts.index)

# spx_call_delta_ts.index = idx
# spx_call_delta_ts.index.name = "symbol"
# print(spx_call_delta_ts)

# Construct two new DataFrames on data used prior, where one contains the time series data and the other contains call characterizing features (expiry, strike)
spx_call_delta_ts = Augmented_SPX_call_delta_series.iloc[:-1, :].copy()
spx_call_delta_ts.columns.name = None
spx_call_info = pd.DataFrame()

idx, spx_call_info["strike"], spx_call_info["expiry"] = zip(*spx_call_delta_ts.index)

spx_call_delta_ts.index = idx
spx_call_delta_ts.index.name = "symbol"
spx_call_delta_ts.columns.name = "date"
spx_call_info.index = idx
spx_call_info.index.name = "symbol"

# Construct a new Series object on data used prior
spx_close_delta_ts = Augmented_SPX_call_delta_series.iloc[-1, :].copy()
spx_close_delta_ts.index.name = "date"
spx_close_delta_ts.rename("close", inplace=True)

# Construct new DataFrame for hedge deltas (Old name: spx_call_hedge_delta_ts)
spx_call_baseline_hedge_delta = (  # DataFrame of hedge delta time series', where each row corresponds to a unique contract ("symbol"), and each column corresponds to a unique date ("date")
    SPX_calls_no_nan
    .pivot(index="symbol", columns="date", values="delta")
    # .reindex(index=contracts)
)
spx_call_baseline_hedge_delta.columns.name = "date"
spx_call_baseline_hedge_delta.index.name = "symbol"

spx_call_baseline_hedge_delta.dropna(inplace=True) # Remove observations (contracts) with incomplete time series (having NaN entries)
# spx_call_hedge_delta_ts_no_nan = spx_call_hedge_delta_ts.dropna() # Remove observations (contracts) with incomplete time series (having NaN entries)

spx_call_baseline_hedge_delta.drop(spx_call_baseline_hedge_delta.columns[-1], axis=1, inplace=True) # Drop last column to ensure compatibility with "spx_call_delta_ts" and "spx_close_delta_ts" (!!)
# ---> Note: "Must go after the dropna() operation, since otherwise the retained indices (contracts), do not match those in spx_call_delta_ts" (!)

assert (spx_call_baseline_hedge_delta.index == spx_call_delta_ts.index).all(), "Indexes (contracts) in spx_call_baseline_hedge_delta do not match those in the SPX call dataframes constructed prior"


## (b) Baseline residuals and SSE
spx_call_baseline_residuals = spx_call_delta_ts.sub(spx_call_baseline_hedge_delta.mul(spx_close_delta_ts, axis=1)) 
# spx_call_baseline_residuals.drop(spx_call_baseline_residuals.columns[-1], axis=1, inplace=True) # Drop last column, which contains only NaN values since the spx call midquote deltas are undefined for t = 2023-02-28 --> Note: is the only column containing NaN's (all 4975 elements), all other columns have 0 NaN values.
# ---> Note: "Operation NO LONGER REQUIRED; since the root problem (dim. incompatibility) that was causing the final column to be full of NaNs, has been identified and resolved." (!)
print(spx_call_baseline_residuals)


## (c) Standardized filters (mandatory)
### Attempt 2
# First filtering step: discard observations with Delta^{BS}_t <= 0.05 at any timestamp t
first_filter = spx_call_baseline_hedge_delta.mask(spx_call_baseline_hedge_delta <= 0.05)
first_filter.dropna(axis=0, inplace=True) ### ??

# Second filtering step: discard observations with Delta^{BS}_t >= 0.05 at any timestamp t
second_filter = first_filter.mask(first_filter >= 0.95)
second_filter.dropna(axis=0, inplace=True) ### ??

# Construct additional DataFrame for the third and final filter, containing D_t (days until maturity), for all contracts (with given expiry) at all times t ("date")
df = SPX_calls_no_nan.copy()
df["days_to_exp"] = (df["exdate"] - df["date"])
spx_call_days_to_exp = (
    df
    .pivot(index="symbol", columns="date", values="days_to_exp")
)

spx_call_days_to_exp = spx_call_days_to_exp.loc[spx_call_baseline_hedge_delta.index]

spx_call_days_to_exp.drop(spx_call_days_to_exp.columns[-1], axis=1, inplace=True) # Drop last column to ensure compatibility with "spx_call_baseline_hedge_delta" (and thus also "spx_call_delta_ts" and "spx_close_delta_ts") (!!)
# ---> Note: "Place this operation after taking a subset of the indices (using .loc), for future safety purposes" (!)

# Third filtering step: discard observations with D_t <= 14 at any timestamp t
third_filter = second_filter.mask(spx_call_days_to_exp.loc[second_filter.index] <= pd.to_timedelta(14, unit= "D"))
third_filter.dropna(axis=0, inplace=True) ### ??

spx_call_baseline_residuals_filtered = spx_call_baseline_residuals.loc[third_filter.index]
spx_call_baseline_sse = spx_call_baseline_residuals_filtered.pow(2).sum(axis=1)


## (d) Standardized bucketing (mandatory)
# Get hedge delta bins
delta_bins = np.linspace(0.05, 0.95, num=10)  # 10 boundaries → 9 bins

# Get time-to-expiry bins
min_days = 14
max_days = spx_call_days_to_exp.max().max().days

tte_bins = np.linspace(min_days, max_days, num=8, dtype=int) # Time-to-expiry bins: 8 boundaries => 7 bins
tte_bins = pd.to_timedelta(tte_bins, unit="D")

# DataFrames / ndarrays to be used
res_df = spx_call_baseline_residuals_filtered
delta_df = spx_call_baseline_hedge_delta.loc[res_df.index]
tte_df = spx_call_days_to_exp.loc[res_df.index]

res_arr = res_df.to_numpy()

# Define bucket labeling
delta_label = []
it = iter(delta_bins)
h1 = next(it)
for _ in range(len(delta_bins)-1):
    h2 = next(it)
    x = f"{round(h1, 2)} < \u0394 <= {round(h2, 2)}"
    delta_label.append(x)
    h1 = h2 

tte_label = []
it = iter(tte_bins)
d1 = next(it)
for _ in range(len(tte_bins)-1):
    d2 = next(it)
    x = f"{d1.days} < D <= {d2.days}"
    tte_label.append(x)
    d1 = d2

# Construct buckets
buckets = {} 

##########TESTTESTTEST
# print(delta_df.shape)
# print(tte_df.shape)
# print("\n", spx_call_delta_ts.shape)
# print(spx_call_baseline_hedge_delta.shape)
# print(spx_close_delta_ts.shape)
# print("\n", spx_call_baseline_residuals_filtered.shape)
##########TESTTESTTEST

for i in range(len(delta_bins)-1):
    h1 = delta_bins[i]
    h2 = delta_bins[i+1]
    row = {}
    row_label = delta_label[i]

    for j in range(len(tte_bins)-1):
        d1 = tte_bins[j]
        d2 = tte_bins[j+1]

        mask = (
            (h1 < delta_df) & (delta_df <= h2) &
            (d1 < tte_df) & (tte_df <= d2)
        )
        mask_arr = mask.to_numpy()

        col_label = tte_label[j]
        row[col_label] = mask_arr
        
    buckets[row_label] = row

# Get global results (SSE/MSE) on the final filtered sample
sse = res_df.pow(2).sum(axis=1)
sse.rename("SSE", inplace=True)
print(sse)

mse = sse / res_df.shape[1]
mse.rename("MSE", inplace=True)
print(mse)

# Get bucket results
# (1) (To be used for heatmaps)
bucket_sse = []

for i in range(len(delta_bins)-1):
    row = []
    row_label = delta_label[i]

    for j in range(len(tte_bins)-1):
        col_label = tte_label[j]

        mask_arr = buckets[row_label][col_label]
        sse = ((mask_arr * res_arr) ** 2).sum()

        row.append(sse)
        
    bucket_sse.append(row)

bucket_sse = np.array(bucket_sse)

# (2)
bucket_sse_per_call = []

for i in range(len(delta_bins)-1):
    row = []
    row_label = delta_label[i]

    for j in range(len(tte_bins)-1):
        col_label = tte_label[j]

        mask_arr = buckets[row_label][col_label]
        sse = ((mask_arr * res_arr) ** 2).sum(axis=1)
        sse = pd.Series(
            sse, 
            index = res_df.index
        )

        row.append(sse)
        
    bucket_sse_per_call.append(row)

bucket_sse_per_call = pd.DataFrame(
    bucket_sse_per_call,
    index = delta_label,
    columns = tte_label,
)

# (3) (To be used for heatmaps)
bucket_mse = [] 

for i in range(len(delta_bins)-1):
    row = []
    row_label = delta_label[i]

    for j in range(len(tte_bins)-1):
        col_label = tte_label[j]

        mask_arr = buckets[row_label][col_label]
        mse = ((mask_arr * res_arr) ** 2).sum() / mask_arr.sum() if mask_arr.sum() > 0 else 0  # Avoids zero-division error (in this case we would not get an error, but rather NaN values, which should in fact be equal to 0 given that mse is calculated for a sample size = 0)

        row.append(mse)
        
    bucket_mse.append(row)

bucket_mse = np.array(bucket_mse)

# (4) The actual results from (1-3)
# (4.1) the results to be exported to csv
print(bucket_sse_per_call)
# ... ADD: csv export-code 
print(bucket_sse)
print(bucket_mse)

# (4.2) heatmap of bucketed sse
fig, ax = plt.subplots()
im = ax.imshow(bucket_sse)

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(len(tte_label)), labels=tte_label,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(delta_label)), labels=delta_label)

# Loop over data dimensions and create text annotations.
for i in range(len(delta_label)):
    for j in range(len(tte_label)):
        text = ax.text(j, i, (bucket_sse / 1000).round(1)[i, j], # With SSE divided by 1000, and then rounded to 1 decimal, to ensure better readability of the heatmap
                       ha="center", va="center", color="w")

ax.set_title("Heatmap of the bucketed SSE (x1000)")
fig.tight_layout()
# plt.show()

# (4.3) heatmap of bucketed mse
fig, ax = plt.subplots()
im = ax.imshow(bucket_mse)

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(len(tte_label)), labels=tte_label,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(delta_label)), labels=delta_label)

# Loop over data dimensions and create text annotations.
for i in range(len(delta_label)):
    for j in range(len(tte_label)):
        text = ax.text(j, i, bucket_mse.round(1)[i, j],  # With MSE rounded to 1 decimal, to ensure better readability of the heatmap
                       ha="center", va="center", color="w")

ax.set_title("Heatmap of the bucketed MSE")
fig.tight_layout()
# plt.show()



###### Compute market-implied volatility + Use this to estimate BS-Delta
#################################### ATTEMPT 1
## "Since we assume the dividens q=0, we can use the following (simplified) method to compute the market-implied volatility, which then yields the estimated Black-Scholes Delta"
# Function definitions below:
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))

def norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2*np.pi)

###### VERSION 2:
def implied_volatility(price_mkt, S, K, tau, r,
                       tol=1e-6, max_iter=100):

    # No-arbitrage bounds (call option)
    intrinsic = max(S - K * np.exp(-r * tau), 0)
    upper_bound = S

    if price_mkt < intrinsic or price_mkt > upper_bound:
        return np.nan, np.nan

    # Vol bounds
    sigma_low = 1e-6
    sigma_high = 5.0   # 500% vol cap (very safe upper limit)

    # Initial guess
    F = S * np.exp(r * tau)
    sigma = np.sqrt(2 * np.pi / tau) * (price_mkt / F)
    sigma = np.clip(sigma, sigma_low, sigma_high)

    for _ in range(max_iter):

        sigma_sqrt_tau = sigma * np.sqrt(tau)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / sigma_sqrt_tau
        d2 = d1 - sigma_sqrt_tau

        price_bs = S * norm_cdf(d1) - K * np.exp(-r * tau) * norm_cdf(d2)
        vega = S * norm_pdf(d1) * np.sqrt(tau)

        diff = price_bs - price_mkt

        if abs(diff) < tol:
            return sigma, norm_cdf(d1)

        # If vega too small → switch to bisection
        if abs(vega) < 1e-8:
            break

        # Newton step
        sigma_new = sigma - diff / vega

        # Keep sigma in bounds
        if sigma_new <= sigma_low or sigma_new >= sigma_high:
            break

        sigma = sigma_new

    # Bisection fallback
    for _ in range(max_iter):

        sigma_mid = 0.5 * (sigma_low + sigma_high)

        sigma_sqrt_tau = sigma_mid * np.sqrt(tau)
        d1 = (np.log(S / K) + (r + 0.5 * sigma_mid**2) * tau) / sigma_sqrt_tau
        d2 = d1 - sigma_sqrt_tau

        price_bs = S * norm_cdf(d1) - K * np.exp(-r * tau) * norm_cdf(d2)

        if abs(price_bs - price_mkt) < tol:
            return sigma_mid, norm_cdf(d1)

        if price_bs > price_mkt:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid

    print("Warning: Max iterations reached without convergence.")
    return np.nan, np.nan


###### VERSION 1:
# def implied_volatility(price_mkt, S, K, tau, r, tol=1e-6, max_iter=100):
#     """
#     Calculate Implied Volatility using Newton-Raphson.
#     Parameters:
#     price_mkt : float - Observed market price of the option
#     S : float - Asset spot price (at time t)
#     K : float - Strike Price
#     tau : float - Year fraction ((T-t)/365)
#     r : float - Risk-free rate
#     Returns:
#     sigma_imp : float - Implied Volatility
#     delta_bs : float - Estimated Black-Scholes Delta
#     """
#     # 1. Initial Guess (Brenner-Subrahmanyam approximation)
#     # This places us close to the solution for ATM options
#     F = S * np.exp(r * tau)
#     sigma = np.sqrt(2 * np.pi / tau) * (price_mkt / F)

#     for i in range(max_iter):
#         # Calculate d1, d2
#         sigma_sqrt_tau = sigma * np.sqrt(tau)
#         d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / sigma_sqrt_tau
#         d2 = d1 - sigma_sqrt_tau

#         # Calculate Price and Vega
#         # Note: We use r=0 formulas here based on lecture assumptions,
#         # but generic implementation would include discount factors.
#         price_bs = S * norm_cdf(d1) - K * np.exp(-r * tau) * norm_cdf(d2)
#         vega = S * norm_pdf(d1) * np.sqrt(tau)

#         # Calculate Error
#         diff = price_bs - price_mkt

#         # Check Convergence
#         if abs(diff) < tol:
#             return sigma, norm_cdf(d1) ###
        
#         # Newton-Raphson Step
#         # Protection against Zero Vega (Deep OTM/ITM)
#         if abs(vega) < 1e-8:
#             print(f"Warning: Vega is zero, Newton method fails at iteration: {i}.")
#             return np.nan, np.nan ###
        
#         sigma = sigma - diff / vega

#     print("Warning: Max iterations reached without convergence.")
#     return sigma, norm_cdf(d1) ###


# Collect necessary data/ parameters
price_mkt = Augmented_SPX_call_midquote_series.iloc[:-1, :].copy()
tau = SPX_call_tau_series.copy()
r = cc_rates.copy()
idx, *_ = zip(*price_mkt.index)

price_mkt.index = idx
tau.index = idx
r.index = idx

price_mkt = price_mkt.loc[third_filter.index]
idx = price_mkt.index.tolist()
cols = price_mkt.columns.tolist()

price_mkt = price_mkt.to_numpy()
S = SPX_close_series.values
K = spx_call_info.loc[third_filter.index]["strike"].values
tau = tau.loc[third_filter.index].to_numpy()
r = r.loc[third_filter.index].to_numpy()


# Compute all the market-implied volatilities + Black-Scholes Deltas
sigma_imp = np.zeros((len(idx), len(cols)))
estim_delta_bs = np.zeros((len(idx), len(cols)))
for i, _ in enumerate(idx):

    for j, _ in enumerate(cols):
        sigma, delta  = implied_volatility(
            price_mkt[i, j], 
            S[j], 
            K[i], 
            tau[i, j], 
            r[i, j]
        )
        sigma_imp[i, j] = sigma
        estim_delta_bs[i, j] = delta


sigma_imp = pd.DataFrame(
    sigma_imp,
    index = idx,
    columns = cols
)
sigma_imp.index.name = "symbol"
sigma_imp.columns.name = "date"

estim_delta_bs = pd.DataFrame(
    estim_delta_bs,
    index = idx,
    columns = cols
)
estim_delta_bs.index.name = "symbol"
estim_delta_bs.columns.name = "date"

print(sigma_imp) ###
print(estim_delta_bs) ###
print(spx_call_baseline_hedge_delta.loc[third_filter.index]) #######


## (f) Hedging residuals and SSEs




## Reporting (5)
# (b/1) Summary statistics of epsilon_t(Delta^{BS})
print("\n", spx_call_baseline_residuals.describe()) # Statistics per date
print(pd.Series(spx_call_baseline_residuals.to_numpy().flatten()).describe()) # Statistics over all values
### (!) ---> Q.: "Should we use the raw residual data: spx_call_baseline_residuals, or the filtered residual data: spx_call_baseline_residuals_filtered?"
### ---> A.: "We should use the unfiltered residual data here; using filtered data only applies to the SSE data."

# (c) Remaining row count after each filter
print("\nRow count after first filter " + r"$(\Delta_t^{\text{BS}})\leq 0.05$" + ": ", first_filter.shape[0])
print("Row count after second filter " + r"$(\Delta_t^{\text{BS}})\geq 0.95$" + ": ", second_filter.shape[0])
print("Row count after third filter " + r"$D_t\leq 14$" + ": ", third_filter.shape[0])

# (b/2) Baseline SSE(Delta^{BS}) after standardized filters
print(spx_call_baseline_sse)

# (b/3) Plot of baseline (hedge) delta vs strike for chosen day and maturity (expiry) slice
repr_date = pd.to_datetime("2023-02-14") # Again we use repr_date = "2023-02-14"
repr_expiry = spx_call_info["expiry"].median()

x = spx_call_info[spx_call_info["expiry"] == repr_expiry]["strike"]
y = spx_call_baseline_hedge_delta.loc[x.index][repr_date]
assert all(x.index == y.index), "Indices (\"symbol\") for x and y in the plot of baseline delta vs strike, do not match"
x = x.values
y = y.values

x_sorted = np.sort(x)
sorted_idx = np.argsort(x)
y_sorted = y[sorted_idx]
plt.figure(figsize=(8, 5))
plt.scatter(x, y, label="Observed values", zorder=3) 
plt.plot(x_sorted, y_sorted, label="Fitted values")
plt.xlabel(r"strike $K$")
plt.ylabel(r"$\Delta_t^{\text{BS}}$", rotation="horizontal")
plt.title(label=r"Plot of $\Delta_t^{\text{BS}}$ vs strike $K$;  " + f"on t={repr_date.date()},  given expiry T={repr_expiry.date()}:")
plt.legend()
plt.grid(True)
# plt.show() # UNCOMMENT THIS in the hand-in doc.
# ---> Note: We should expect the (baseline) hedge delta to decrease as the strike K increases --> ensures delta hedge is a "good" substitute for moneyness = S_0/K

# (b/4) Study of the misspecification effect on the hedging performance on the implied-volatility parameter
# ... ToDo!!
