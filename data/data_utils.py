import pandas as pd


def resample_candles(df, timeframe):
    df.index = pd.to_datetime(df.Date)
    return df.resample(timeframe).agg({
        'Open': 'first',
        'Close': 'last',
        'High': 'max',
        'Low': 'min',
        'Volume': 'sum'
    }).dropna()

def prepare_data(bars_df, timeframe):
    bars_df = bars_df.rename(columns={'open_price': 'Open', 'close_price': 'Close', 'high_price': 'High', 'low_price': 'Low', 'close_time': 'Date', 'volume': 'Volume'})
    bars_df = resample_candles(bars_df, timeframe)
    bars_df['Date_dt'] = pd.to_datetime(bars_df.index)
    bars_df = bars_df.reset_index(drop=True)
    return bars_df