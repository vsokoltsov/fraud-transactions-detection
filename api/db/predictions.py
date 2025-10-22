import pandas as pd
import numpy as np

EPS = 1e-9

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = tx_amount_columns(df)
    df = time_since_prev_trx(df)
    df = burst_deviation(df)
    df = rs_anomaly(df)
    df = hour_deviates(df)
    df = robust_amount_outlier(df)
    df = day_of_week_zscore_outlier(df)
    df = fraud_burst_candidate(df)
    return df


def tx_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['tx_amount_log'] = np.log1p(df['tx_amount'])
    df['tx_amount_log_mean'] = df.groupby('customer_id')['tx_amount_log'].transform('mean')
    df['tx_amount_log_std'] = df.groupby('customer_id')['tx_amount_log'].transform('std')
    df['tx_amount_log_deviates'] = (
        (df['tx_amount_log'] < (df['tx_amount_log_mean'] - df['tx_amount_log_std'])) | 
        (df['tx_amount_log'] > (df['tx_amount_log_mean'] + df['tx_amount_log_std']))
    ).astype(int)
    return df

def time_since_prev_trx(df: pd.DataFrame) -> pd.DataFrame:
    df['secs_since_prev_tx'] = (
        df.groupby('customer_id')['tx_datetime']
        .diff()
        .dt.total_seconds()
        .fillna(-1)
    )
    return df

def burst_deviation(df: pd.DataFrame) -> pd.DataFrame:
    df['burst_id'] = df.groupby('customer_id')['secs_since_prev_tx'].transform(lambda x: (x > 3600).cumsum())
    df['n_tx_in_burst'] = df.groupby(['customer_id', 'burst_id'])['tx_amount_log'].transform('count')
    df['burst_mean'] = df.groupby('customer_id')['n_tx_in_burst'].transform('mean').fillna(0)
    df['burst_std'] = df.groupby('customer_id')['n_tx_in_burst'].transform('std').fillna(0)
    df['n_trx_per_burst_deviates'] = (
        (df['n_tx_in_burst'] < (df['burst_mean'] - df['burst_std'])) | 
        (df['n_tx_in_burst'] > (df['burst_mean'] + df['burst_std']))
    ).astype(int)
    return df

def rs_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    df['zscore'] = (df['tx_amount_log'] - df.groupby('customer_id')['tx_amount_log'].transform('mean')) / df.groupby('customer_id')['tx_amount_log'].transform('std')
    df['is_zscore_outlier'] = (df['zscore'] > 3).astype(int)
    q1 = df.groupby('customer_id')['tx_amount_log'].transform(lambda s: s.quantile(0.25))
    q3 = df.groupby('customer_id')['tx_amount_log'].transform(lambda s: s.quantile(0.75))
    med = df.groupby('customer_id')['tx_amount_log'].transform('median')
    iqr = (q3 - q1).replace(0, np.nan).fillna(EPS)
    df['is_iqr_outlier'] = (~df['tx_amount_log'].between(q1 - 0.5*iqr, q3 + 0.5*iqr)).astype(int)

    df['tx_amount_log_scaled'] = (df['tx_amount_log'] - med) / iqr
    df['is_rs_anomaly'] = (df['tx_amount_log_scaled'].abs() > 1.5).astype(int)
    return df

def hour_deviates(df: pd.DataFrame) -> pd.DataFrame:
    df['day_of_week'] = df['tx_datetime'].dt.day_of_week
    df['hour'] = df['tx_datetime'].dt.hour
    df['month'] = df['tx_datetime'].dt.month
    df['is_month_start'] = df['tx_datetime'].dt.is_month_start.astype('int')
    df['is_month_end'] = df['tx_datetime'].dt.is_month_end.astype('int')
    df['is_weekend'] = df['day_of_week'].isin({5, 6}).astype('int')
    df['hour_zscore'] = (df['hour'] - df.groupby('customer_id')['hour'].transform('mean')) / df.groupby('customer_id')['hour'].transform('std')
    df['hour_zscore_deviates'] = (df['hour_zscore'].abs() > 2).astype(int)
    return df

def robust_amount_outlier(df: pd.DataFrame) -> pd.DataFrame:
    window = 20
    min_periods = 5

    df = df.sort_values(['customer_id', 'tx_datetime']).copy()
    g = df.groupby('customer_id', sort=False)
    r5 = g['tx_amount_log'].rolling(window=window, min_periods=min_periods)
    r1 = g['tx_amount_log'].rolling(window=window, min_periods=1)

    med5 = r5.median().reset_index(level=0, drop=True)
    q1_5 = r5.quantile(0.25).reset_index(level=0, drop=True)
    q3_5 = r5.quantile(0.75).reset_index(level=0, drop=True)

    med1 = r1.median().reset_index(level=0, drop=True)
    q1_1 = r1.quantile(0.25).reset_index(level=0, drop=True)
    q3_1 = r1.quantile(0.75).reset_index(level=0, drop=True)

    df['rolling_median'] = med5.fillna(med1)
    df['q1'] = q1_5.fillna(q1_1)
    df['q3'] = q3_5.fillna(q3_1)

    df['iqr'] = df['q3'] - df['q1']

    df['amount_robust_rolling20'] = (df['tx_amount_log'] - df['rolling_median']) / (df['iqr'] + EPS)

    df['amount_robust_rolling20'] = df['amount_robust_rolling20'].fillna(0)
    df['is_amount_robust_rolling_outlier'] = (df['amount_robust_rolling20'] > 3).astype(int)
    return df

def day_of_week_zscore_outlier(df: pd.DataFrame) -> pd.DataFrame:
    mean_ = df.groupby('customer_id')['day_of_week'].transform('mean')
    std_ = df.groupby('customer_id')['day_of_week'].transform('std')
    df['day_of_week_mean'] = mean_.fillna(0)
    df['day_of_week_std'] = std_.fillna(0)
    df['is_day_of_week_mean_outlier'] = (
        (df['day_of_week'] < (df['day_of_week_mean'] - df['day_of_week_std'])) |
        (df['day_of_week'] > (df['day_of_week_mean'] + df['day_of_week_std']))
    ).astype(int)
    df['day_of_week_zscore'] = (df['day_of_week'] - df.groupby('customer_id')['day_of_week'].transform('mean')) / df.groupby('customer_id')['day_of_week'].transform('std')
    df['is_day_of_week_zscore_outlier'] = (df['day_of_week_zscore'].abs() > 2).astype(int)
    return df

def fraud_burst_candidate(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.sort_values(['customer_id','tx_datetime']).reset_index(drop=True).copy()
    dfc['n_tx_in_prev_24h'] = 0
    one_day = np.timedelta64(1, 'D')

    for cid, grp in dfc.groupby('customer_id'):
        t = grp['tx_datetime'].to_numpy('datetime64[ns]')
        left = t - one_day
        j = np.searchsorted(t, left, side='left')
        cnt = np.arange(len(t)) - j
        dfc.loc[grp.index, 'n_tx_in_prev_24h'] = cnt.astype(int)

    g = dfc.groupby('customer_id')
    dfc['q90_prev'] = g['n_tx_in_prev_24h'].transform(
        lambda s: s.shift(1).expanding().quantile(0.90).fillna(0)
    )
    dfc['is_24h_burst'] = (
        dfc['n_tx_in_prev_24h'] >= dfc['q90_prev']
    ).fillna(False).astype(int)
    dfc['is_24h_burst_fixed'] = (dfc['n_tx_in_prev_24h'] >= 3).astype(int)
    dfc['day'] = dfc['tx_datetime'].dt.date
    dg = dfc.groupby(['customer_id','day'])['tx_amount_log']

    day_median = dg.transform('median')
    day_mad = dg.transform(lambda s: (s - s.median()).abs().median())
    dfc['z_in_day_robust'] = (dfc['tx_amount_log'] - day_median) / (1.4826*day_mad + EPS)

    dfc['is_anomalous_in_day'] = (dfc['z_in_day_robust'].abs() > 2.5).astype(int)
    dfc['fraud_burst_candidate'] = (
        (dfc['is_24h_burst'] == 1) &
        (dfc['z_in_day_robust'] > 1.0)
    ).astype(int)
    return dfc