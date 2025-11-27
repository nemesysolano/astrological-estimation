import pandas as pd
import numpy as np

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

def add_closest_higher_high(historical_data):
    highs = historical_data['High']
    num_rows = len(historical_data)
    result_val = np.full(num_rows, np.nan)
    result_days = np.full(num_rows, np.nan)

    for t in range(1, num_rows):
        current_high = highs.iloc[t]
        for i in range(1, t + 1):
            past_high = highs.iloc[t - i]
            if past_high > current_high:
                result_val[t] = past_high
                result_days[t] = i
                break
    historical_data['h_↑'] = result_val
    historical_data['Dh_↑'] = result_days
    historical_data.dropna(inplace=True)

def add_closest_lower_high(historical_data):
    highs = historical_data['High']
    num_rows = len(historical_data)
    result_val = np.full(num_rows, np.nan)
    result_days = np.full(num_rows, np.nan)

    for t in range(1, num_rows):
        current_high = highs.iloc[t]
        for i in range(1, t + 1):
            past_high = highs.iloc[t - i]
            if past_high < current_high:
                result_val[t] = past_high
                result_days[t] = i
                break
    historical_data['h_↓'] = result_val
    historical_data['Dh_↓'] = result_days
    historical_data.dropna(inplace=True)

def add_closest_higher_low(historical_data):
    lows = historical_data['Low']
    num_rows = len(historical_data)
    result_val = np.full(num_rows, np.nan)
    result_days = np.full(num_rows, np.nan)

    for t in range(1, num_rows):
        current_low = lows.iloc[t]
        for i in range(1, t + 1):
            past_low = lows.iloc[t - i]
            if past_low > current_low:
                result_val[t] = past_low
                result_days[t] = i
                break
    historical_data['l_↑'] = result_val
    historical_data['Dl_↑'] = result_days
    historical_data.dropna(inplace=True)

def add_closest_lower_low(historical_data):
    lows = historical_data['Low']
    num_rows = len(historical_data)
    result_val = np.full(num_rows, np.nan)
    result_days = np.full(num_rows, np.nan)
    for t in range(1, num_rows):
        current_low = lows.iloc[t]
        for i in range(1, t + 1):
            past_low = lows.iloc[t - i]
            if past_low < current_low:
                result_val[t] = past_low
                result_days[t] = i
                break
    historical_data['l_↓'] = result_val
    historical_data['Dl_↓'] = result_days
    historical_data.dropna(inplace=True)

def identify_pivots(df, window=5):
    df['is_pivot_high'] = df['High'].rolling(window=window*2+1, center=True).max() == df['High']
    df['is_pivot_low'] = df['Low'].rolling(window=window*2+1, center=True).min() == df['Low']
    return df

def get_nearest_structural_extreme(current_idx, current_val, df, pivot_col, price_col, comparison):    
    for past_idx in range(current_idx - 1, -1, -1):
        if df.iat[past_idx, df.columns.get_loc(pivot_col)]: # Check if it is a pivot
            past_val = df.iat[past_idx, df.columns.get_loc(price_col)]
            if comparison(past_val, current_val):
                return past_idx
    return -1

def add_cosine_and_sine_for_price_time_angles(df):
    """
    Calculates Price-Time angles based on Fractal Pivots and ATR Normalization.
    
    Implements:
    A. Fractal Pivot Search (Structural points instead of noise).
    B. ATR Normalization (Gann Box stabilization).
    C. Edge Case Handling (Breakout constants).
    """
    
    # 1. Ensure Dependencies
    # We need ATR for normalization. Assuming 'ATR' or 'Atrp14' exists. 
    # If using 'Atrp14' (percentage), we convert back to absolute ATR approx or calculate it.
    # Here we will calculate a standard 14-period ATR for safety.
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()

    # 2. Identify Structural Pivots (Suggestion A)
    # Using a 5-day window (2 days before, 2 days after)
    df = identify_pivots(df, window=2)

    # Initialize columns for angles (theta)
    # 1: Higher High, 2: Lower High, 3: Higher Low, 4: Lower Low
    thetas = {1: [], 2: [], 3: [], 4: []}
    
    # Comparisons for the 4 extremes
    # Higher High: Past High > Current High
    # Lower High: Past High < Current High
    # Higher Low: Past Low > Current Low
    # Lower Low: Past Low < Current Low
    comparisons = {
        1: ('is_pivot_high', 'High', lambda p, c: p > c),
        2: ('is_pivot_high', 'High', lambda p, c: p < c),
        3: ('is_pivot_low', 'Low', lambda p, c: p > c),
        4: ('is_pivot_low', 'Low', lambda p, c: p < c)
    }

    # Iterate through the DataFrame
    # Note: Iterating rows is slow in Pandas, but necessary for complex lookback logic 
    # that varies per row.
    
    for i in range(len(df)):
        if i < 20: # Skip beginning where ATR/Pivots might be unstable
            for k in thetas: thetas[k].append(0)
            continue
            
        atr = df.iat[i, df.columns.get_loc('ATR_14')]
        if pd.isna(atr) or atr == 0:
            atr = df.iat[i, df.columns.get_loc('Close')] * 0.01 # Fallback
            
        current_time_idx = i
        
        for k, (pivot_col, price_col, comp_func) in comparisons.items():
            current_price = df.iat[i, df.columns.get_loc(price_col)]
            
            # Find closest structural extreme
            past_idx = get_nearest_structural_extreme(
                current_time_idx, current_price, df, pivot_col, price_col, comp_func
            )
            
            if past_idx != -1:
                # Suggestion B: Stabilized Normalization
                # Slope = (Delta Price) / (Delta Time * ATR)
                # This normalizes the "speed" of the move relative to volatility.
                
                past_price = df.iat[past_idx, df.columns.get_loc(price_col)]
                delta_price = current_price - past_price
                delta_time = current_time_idx - past_idx # Number of bars
                
                # Gann Theory: 45 degrees (slope 1) is a "balanced" market.
                # If price moves 1 ATR in 1 day, slope is 1.
                normalized_slope = delta_price / (delta_time * atr)
                
                # Calculate Angle in Radians
                angle = np.arctan(normalized_slope)
                thetas[k].append(angle)
            else:
                # Suggestion C: Edge Case Handling (Breakout)
                # If we are making a New High (no Higher High found), 
                # we are in "blue sky" mode. 
                # Resistance is effectively infinite or vertical (90 deg / pi/2).
                if k == 1: # Higher High (Resistance) missing -> Bullish Breakout
                    thetas[k].append(np.pi / 2)
                elif k == 4: # Lower Low (Support) missing -> Bearish Breakdown
                    thetas[k].append(-np.pi / 2)
                else:
                    thetas[k].append(0)


    # 3. Add Cosine and Sine features
    for j in range(len(thetas)):
        k = j + 1
        df[f'cos_θ{k}'] = np.cos(thetas[k])
        df[f'sin_θ{k}'] = np.sin(thetas[k])


    # Clean up temporary columns
    df.drop(columns=['is_pivot_high', 'is_pivot_low', 'ATR_14'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    return df