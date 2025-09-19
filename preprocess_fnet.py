import os, glob, re
import numpy as np
import pandas as pd
import torch


def _year_from_filepath(path: str) -> int:
    """
    Extract YYYY from filenames like '20110908-223754-UsAzAsu702.txt'
    """
    base = os.path.basename(path)
    m = re.match(r'^(?P<yyyymmdd>\d{8})-\d{6}-', base)
    if not m:
        raise ValueError(f"Cannot parse date from filename: {base}")
    yyyymmdd = m.group('yyyymmdd')
    return int(yyyymmdd[:4])

def preprocess_one_raw(df_raw: pd.DataFrame, source_path: str) -> pd.DataFrame:
    """
    Input:
        df_raw: read via pd.read_csv(file, sep=' ', header=None)
                last column is 'Frequency\\tDelay'
        source_path: full file path, used to extract YEAR from filename
    Output:
        tidy df with columns:
        ['DateTime','t','Frequency','Delay','r','roc_ps']
        (and attrs['dt'] = sampling interval in seconds)
    """
    df = df_raw.copy()

    # keep only the last two raw columns: [time_index, 'Frequency\\tDelay']
    if df.shape[1] > 2:
        df = df.iloc[:, -2:].copy()
    df.columns = ['time', 'value']

    # split 'Frequency\\tDelay'
    parts = df['value'].astype(str).str.split(r'\t', n=1, expand=True)
    df['Frequency'] = parts[0].astype(float)
    df['Delay'] = pd.to_numeric(parts[1], errors='coerce')
    df.drop(columns=['value'], inplace=True)

    # ---- decode the FNET "time index" to Month/Day/Hour/Minute/Second ----
    # time index = Month*1e7 + Day*1e5 + Hour*3600 + Minute*60 + Second
    v = df['time'].astype(float).to_numpy()

    month = (v // 1e7).astype(int)
    rem1  = v - month * 1e7
    day   = (rem1 // 1e5).astype(int)
    rem2  = rem1 - day * 1e5
    hour  = (rem2 // 3600).astype(int)
    rem3  = rem2 - hour * 3600
    minute = (rem3 // 60).astype(int)
    second = (rem3 - minute * 60)  # may contain tenths (float)

    # Build absolute DateTime using YEAR from filename and M/D/H/M/S from the index
    year = _year_from_filepath(source_path)

    # midnight for each row (handles month/day changes if present)
    ymd = (year * 10000 + month * 100 + day).astype(int)
    midnight = pd.to_datetime(ymd, format='%Y%m%d')

    # seconds since midnight (float supports fractional seconds)
    total_deci = (hour * 36000 + minute * 600 + np.rint(second * 10).astype('int64'))
    dt_series = midnight + pd.to_timedelta(total_deci * 100_000_000, unit='ns')

    df['DateTime'] = dt_series

    # relative seconds from the first sample (robust across minute/hour boundaries)
    df['t'] = (df['DateTime'] - df['DateTime'].iloc[0]).dt.total_seconds()

    # compute sampling interval from absolute time
    dt_vals = df['DateTime'].diff().dt.total_seconds().dropna()
    dt = float(np.median(dt_vals)) if len(dt_vals) else np.nan
    df.attrs['dt'] = dt

    # physics-aware residuals
    df['r'] = df['Frequency'] - 60.0
    df['roc_ps'] = df['r'].diff().fillna(0.0) / (dt if np.isfinite(dt) and dt > 0 else 1.0)

    # return tidy columns
    return df[['DateTime', 't', 'Frequency', 'Delay', 'r', 'roc_ps']]

def site_id_from_path(path: str) -> str:
    """e.g., '.../20110908-223754-UsAzAsu702.txt' -> 'UsAzAsu702'"""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    # Drop the leading datetime if present
    m = re.match(r'^\d{8}-\d{6}-(.+)$', stem)
    return m.group(1) if m else stem

def load_day_folder(path_data: str):
    """
    Returns:
        sites: [site_id,...] in a fixed order
        data:  {site_id: tidy_df}
        dt:    sampling interval (assumed same for all)
    """
    files = sorted(glob.glob(os.path.join(path_data, '*.txt')))
    data, sites, dt_list, date_list = {}, [], [], []
    for f in files:
        df_raw = pd.read_csv(f, sep=" ", header=None, engine='python')
        df = preprocess_one_raw(df_raw, source_path=f)
        if len(df) < 1000:
            print(f"[skip] {f} has only {len(df_raw)} rows")
            continue
        sid = site_id_from_path(f)
        data[sid] = df
        sites.append(sid)
        dt_list.append(df.attrs.get('dt', np.nan))    
    dt = float(np.nanmedian(dt_list))
    return sites, data, dt


def estimate_event_time(data: dict) -> float:
    """
    data: {site_id: df with columns ['t', 'r', 'roc_ps']}
    Returns global event onset time (seconds from start).
    """
    times = []
    onset_percentile = 15.0
    for sid, df in data.items():
        idx = df['roc_ps'].abs().idxmax()
        times.append(float(df.loc[idx, 't']))
    return float(np.percentile(times, onset_percentile))