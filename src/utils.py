def fetch_spx_close(start_date, end_date):
    """
    Fetch an SPX-like daily close series.
    Primary: FRED SP500
    Fallback: Stooq ˆSPX
    Returns a DataFrame with a ’Close’ column and DatetimeIndex.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Try FRED first
    try:
        df = pdr.DataReader("SP500", "fred", start, end)
        df = df.rename(columns={"SP500": "Close"}).dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        pass

    # Fallback: Stooq
    df = pdr.DataReader("ˆSPX", "stooq", start, end)
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    return df[["Close"]].dropna()

def fetch_vix_close(start_date, end_date):
    """
    Fetch daily VIX close series.
    Primary: FRED VIXCLS Fallback: Stooq VI.F (VIX)
    Returns a DataFrame with a ’Close’ column and DatetimeIndex.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Try FRED first
    try:
        df = pdr.DataReader("VIXCLS", "fred", start, end)
        df = df.rename(columns={"VIXCLS": "Close"}).dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        pass

    # Fallback: Stooq (try common symbols)
    for sym in ["VI.F", "vi.f", "ˆVIX", "ˆvix"]:
        try:
            df = pdr.DataReader(sym, "stooq", start, end)
            df = df.sort_index()
            df.index = pd.to_datetime(df.index)
            if "Close" in df.columns and not df.empty:
                return df[["Close"]].dropna()
        except Exception:
            continue

    raise ValueError("No VIX data found from FRED or Stooq.")
