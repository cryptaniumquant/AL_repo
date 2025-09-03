import time
import numpy as np

def trailing_step_to_price_value(df, position, prev_value, step_pct,
                                 allow_step_back, consider_color, current_bar):

    df = df.iloc[current_bar]
    if position.IsLong:
        distance = df['Low'] - prev_value
        step = distance * (step_pct * 0.01)
        new_value = prev_value
        if (df['Close'] > df['Open']) or (not consider_color):
            new_value = prev_value + step
        if not allow_step_back:
            new_value = max(prev_value, new_value)

    else:
        distance = prev_value - df['High']
        step = distance * (step_pct * 0.01)
        new_value = prev_value
        if (df['Close'] < df['Open']) or (not consider_color):
            new_value = prev_value - step
        if not allow_step_back:
            new_value = min(prev_value, new_value)

    return new_value


def trailing_atr_percent(df, position, prev_value, atr_series, atr_percent, current_bar, allow_step_back):

    current_bar_data = df.iloc[current_bar]
    current_atr = atr_series[current_bar]
    step = current_atr * (atr_percent * 0.01)

    if position.IsLong:
        move_condition = current_bar_data['Close'] > current_bar_data['Open']
        new_value = prev_value + step if move_condition else prev_value

        if not allow_step_back:
            new_value = max(prev_value, new_value)

    else:
        move_condition = current_bar_data['Close'] < current_bar_data['Open']
        new_value = prev_value - step if move_condition else prev_value

        if not allow_step_back:
            new_value = min(prev_value, new_value)

    return new_value

def trailing_volatility_time_based(df, position, stop_price_long_pd, volatility_series,
                                   volatility_percent, current_bar, minutes_interval=60):

    entry_time = (df.iloc[position.EntryBarNum])['Date_dt']
    current_time = df.iloc[current_bar]['Date_dt']
    time_elapsed = current_time - entry_time

    total_minutes = int(time_elapsed.total_seconds() // minutes_interval)
    steps = total_minutes // minutes_interval

    initial_stop = stop_price_long_pd[position.EntryBarNum-1]
    if steps <= 0:
        return initial_stop

    current_atr = volatility_series[current_bar]
    step_size = current_atr * (volatility_percent * 0.01)
    total_shift = step_size * steps

    new_stop = initial_stop + total_shift if position.IsLong else initial_stop - total_shift

    return new_stop
