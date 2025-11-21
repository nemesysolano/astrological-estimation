import pandas as pd
import yfinance as yf
import os
import re
import numpy as np
from typing import Callable, Union

def read_csv(path):
    historical_data = pd.read_csv(path, parse_dates=True, date_format='%Y-%m-%d ', index_col='Date')
    return historical_data

def remove_timezone_from_json_dates(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        content = f.read()

    modified_content = re.sub(r'00:00:00-0\d:00\s?', '', content).replace(" ,", ",")

    with open(file_path, 'w') as f:
        f.write(modified_content)

def add_relative_volume(ticker, historical_data):
    market_cap = ticker.info.get('marketCap')
    historical_data['relative_volume'] = historical_data['Volume'] / (market_cap / historical_data['Close'])    
    historical_data.dropna(inplace=True)

def add_fast_trend_run(historical_data):
    close_prices = historical_data['Close']
    diffs = close_prices.diff().dropna()

    if len(diffs) < 1:
        historical_data['fast_trend_run'] = np.nan
        return

    fast_trend_run = np.zeros(len(historical_data))
    fast_trend_run[0] = np.nan  # No trend run for the first element

    sum_diff = diffs.iloc[0]
    fast_trend_run[1] = sum_diff
    sign = np.sign(sum_diff)

    for i in range(1, len(diffs)):
        d = diffs.iloc[i]
        if np.sign(d) == sign or sign == 0:
            sum_diff += d
        else:
            sum_diff = d
            sign = np.sign(d)
        fast_trend_run[i+1] = sum_diff

    historical_data['fast_trend_run'] = fast_trend_run
    historical_data.dropna(inplace=True)

def add_structural_direction(historical_data):
    high_prices = historical_data['High']
    low_prices = historical_data['Low']
    num_rows = len(historical_data)
    structural_direction = np.full(num_rows, np.nan)

    for i in range(1, num_rows):
        h_t = high_prices.iloc[i]
        h_t_minus_1 = high_prices.iloc[i-1]
        l_t = low_prices.iloc[i]
        l_t_minus_1 = low_prices.iloc[i-1]

        if h_t > h_t_minus_1 and l_t >= l_t_minus_1:
            structural_direction[i] = 1

        elif l_t < l_t_minus_1 and h_t <= h_t_minus_1:
            structural_direction[i] = -1

        else:
            structural_direction[i] = structural_direction[i-1]

    historical_data['structural_direction'] = structural_direction
    historical_data.dropna(inplace=True)

def add_slow_trend_run(historical_data):
    close_prices = historical_data['Close']
    structural_directions = historical_data['structural_direction']
    num_rows = len(historical_data)

    slow_trend_run = np.full(num_rows, np.nan)
    t_s = -1  # Start index of the slow trend

    for i in range(1, num_rows):
        sd_t = structural_directions.iloc[i]
        sd_t_minus_1 = structural_directions.iloc[i-1]

        # A new slow trend starts when the structural direction changes.
        # We also need to handle the initial NaN values.
        if sd_t != sd_t_minus_1 and not np.isnan(sd_t):
            t_s = i

        # Calculate slow trend run if t_s is set and we can access c_{t_s-1}
        if t_s > 0:
            slow_trend_run[i] = close_prices.iloc[i] - close_prices.iloc[t_s - 1]

    historical_data['slow_trend_run'] = slow_trend_run
    historical_data.dropna(inplace=True)

def add_breaking_gap(historical_data):
    high_prices = historical_data['High']
    low_prices = historical_data['Low']
    slow_trend = historical_data['slow_trend_run']
    num_rows = len(historical_data)

    breaking_gap = np.full(num_rows, np.nan)
    for i in range(2, num_rows):
        gap = 0.0
        slow_trend_t_minus_1 = slow_trend.iloc[i-1]

        if slow_trend_t_minus_1 > 0:
            if low_prices.iloc[i] < low_prices.iloc[i-2]:
                gap = low_prices.iloc[i-2] - low_prices.iloc[i]

        elif slow_trend_t_minus_1 < 0:
            if high_prices.iloc[i] > high_prices.iloc[i-2]:
                gap = high_prices.iloc[i] - high_prices.iloc[i-2]

        breaking_gap[i] = gap

    historical_data['breaking_gap'] = breaking_gap
    historical_data.dropna(inplace=True)

def add_fast_swing_ratio(historical_data):
    breaking_gap = historical_data['breaking_gap']
    fast_trend_run = historical_data['fast_trend_run']

    ratio = np.divide(
        breaking_gap,
        np.abs(fast_trend_run),
        out=np.full_like(breaking_gap, np.nan),
        where=(fast_trend_run != 0)
    )

    fast_swing_ratio = np.minimum(2, np.square(ratio))
    historical_data['fast_swing_ratio'] = fast_swing_ratio
    historical_data.dropna(inplace=True)

def last_opposite_to_slow_run(slow_trend_runs, t):
    if t >= len(slow_trend_runs):
        return np.nan

    current_run = slow_trend_runs.iloc[t]
    if np.isnan(current_run) or current_run == 0:
        return np.nan

    current_sign = np.sign(current_run)

    for i in range(t - 1, -1, -1):
        previous_run = slow_trend_runs.iloc[i]
        if not np.isnan(previous_run) and np.sign(previous_run) == -current_sign:
            return previous_run

    return np.nan

def add_slow_swing_ratio(historical_data):
    pass
    slow_trend_runs = historical_data['slow_trend_run']
    num_rows = len(historical_data)
    slow_swing_ratio_values = np.full(num_rows, np.nan)

    for i in range(num_rows):
        r_star_s_t = last_opposite_to_slow_run(slow_trend_runs, i)
        if not np.isnan(r_star_s_t) and r_star_s_t != 0:
            ratio = slow_trend_runs.iloc[i] / np.abs(r_star_s_t)
            slow_swing_ratio_values[i] = min(2.0, ratio**2)

    historical_data['slow_swing_ratio'] = slow_swing_ratio_values
    historical_data.dropna(inplace=True)

def add_directional_probabilities(historical_data) -> pd.DataFrame:    
    df = historical_data.copy()

    # Conditions for trend alignment
    aligned_trends = np.sign(df['fast_trend_run']) == np.sign(df['slow_trend_run'])
    conflicting_trends = ~aligned_trends

    # Aligned Trends Calculation
    # Both trends are ascending
    aligned_up = aligned_trends & (df['fast_trend_run'] > 0)
    # Both trends are descending
    aligned_down = aligned_trends & (df['fast_trend_run'] < 0)

    # Conflicting Trends Calculation
    # Fast ascending, slow descending
    conflicting_fast_up = conflicting_trends & (df['fast_trend_run'] > 0)
    # Fast descending, slow ascending
    conflicting_fast_down = conflicting_trends & (df['fast_trend_run'] < 0)

    # Initialize probability columns
    df['p_up'] = 0.0
    df['p_down'] = 0.0

    # --- Aligned Trends ---
    # Sum of swing ratios for aligned trends
    s_sum = df.loc[aligned_trends, 'slow_swing_ratio'] + df.loc[aligned_trends, 'fast_swing_ratio']
    prob_aligned = np.minimum(1, np.abs(s_sum) / 2)

    # Assign probabilities for aligned trends
    df.loc[aligned_up, 'p_up'] = prob_aligned[aligned_up]
    df.loc[aligned_up, 'p_down'] = 1 - df.loc[aligned_up, 'p_up']

    df.loc[aligned_down, 'p_down'] = prob_aligned[aligned_down]
    df.loc[aligned_down, 'p_up'] = 1 - df.loc[aligned_down, 'p_down']

    # --- Conflicting Trends ---
    # Ratio for conflicting trends
    s_f = df.loc[conflicting_trends, 'fast_swing_ratio']
    s_s = df.loc[conflicting_trends, 'slow_swing_ratio']    
    s_sum_conflicting = s_f + s_s    
    
    # Calculate probability for conflicting trends, with a default of 0.5
    prob_conflicting = pd.Series(0.5, index=s_f.index)
    
    # Apply the formula only where sf > 0 and the sum is not zero
    mask = (s_f > 0) & (s_sum_conflicting != 0)
    prob_conflicting[mask] = s_f[mask] / s_sum_conflicting[mask]

    # Assign probabilities for conflicting trends
    df.loc[conflicting_fast_up, 'p_up'] = prob_conflicting[conflicting_fast_up]
    df.loc[conflicting_fast_up, 'p_down'] = 1 - df.loc[conflicting_fast_up, 'p_up']

    df.loc[conflicting_fast_down, 'p_down'] = prob_conflicting[conflicting_fast_down]
    df.loc[conflicting_fast_down, 'p_up'] = 1 - df.loc[conflicting_fast_down, 'p_down']

    return df

def import_market_data(symbol):
    module_dir = os.path.dirname(__file__)
    data_dir = os.path.join(module_dir, 'data')
    output_path = os.path.join(data_dir, f"{symbol}.csv")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(output_path):
        ticker = yf.Ticker(symbol)        
        historical_data = ticker.history(period="5y", interval="1d")  
        historical_data.to_csv(output_path)        

        remove_timezone_from_json_dates(output_path)
        historical_data = pd.read_csv(output_path, parse_dates=True, date_format='%Y-%m-%d', index_col='Date')

        add_relative_volume(ticker, historical_data)
        add_fast_trend_run(historical_data) # input for breaking_gap
        add_structural_direction(historical_data) # Input for slow_trend_run
        add_slow_trend_run(historical_data) # input for breaking_gap
        add_breaking_gap(historical_data) # input fast_swing_ratio
        add_fast_swing_ratio(historical_data) #input for directional probabilities
        add_slow_swing_ratio(historical_data) #input for directional probabilities
        historical_data = add_directional_probabilities(historical_data)

        historical_data.to_csv(output_path)        

def load_market_data(symbol):
    module_dir = os.path.dirname(__file__)
    data_dir = os.path.join(module_dir, 'data')
    input_path = os.path.join(data_dir, f"{symbol}.csv")
    return pd.read_csv(input_path, parse_dates=True, date_format='%Y-%m-%d', index_col='Date')
