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
print(SPX_calls_no_nan)
# print(Augmented_SPX_call_midquote_series)

df = Augmented_SPX_call_midquote_series.iloc[:-1, :].copy()
df.columns.name = None
idx, df["strike"], df["expiry"] = zip(*df.index)

df.index = idx
df.index.name = "symbol"
print(df)

SPX_mq_delta_hedge = df.merge(SPX_calls_no_nan[["symbol", "delta"]], how="left", left_index=True, right_on="symbol")
# Note(!!): There seem to be multiple delta entries per Contract, hence may need to create a tuple (mq, delta_hedge) for each date
SPX_mq_delta_hedge.set_index("symbol", inplace=True)
print(SPX_mq_delta_hedge)


## (b) Baseline residuals and SSE


## (c) Standardized filters (mandatory)


## (d) Standardized bucketing (mandatory)


## (f) Hedging residuals and SSEs
