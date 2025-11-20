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

def add_trend_info_column(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates two sets of run metrics:
    1. Directional Run (R(t), n): Resets on every close price direction change (c[t] != c[t-1]).
       This is the fast-resetting metric required for the Swing Ratio S(t).
    2. Structural Run (R_struct, n_struct): Resets only when the structural trend_status
       (based on l[t-3]/h[t-3]) changes (from 1 to -1/0 or -1 to 1/0).

    The 'trend_status' column indicates:
    - 1: Ascending Trend is holding (l[t] >= l[t-3])
    - -1: Descending Trend is holding (h[t] <= h[t-3])
    - 0: Neither trend is holding (consolidation or violation)

    Args:
        data_frame: A pandas DataFrame containing 'Close', 'Low', and 'High'
                    price columns, indexed chronologically.

    Returns:
        The DataFrame with all five calculated trend columns added.
    """
    index = data_frame.index
    length = len(index)

    # --- Helper functions to access data by index (t) ---
    c: Callable[[int], float] = lambda t: data_frame.loc[index[t], 'Close']
    l: Callable[[int], float] = lambda t: data_frame.loc[index[t], 'Low']
    h: Callable[[int], float] = lambda t: data_frame.loc[index[t], 'High']

    # --- Trend Variables Initialization ---
    trend_status = np.zeros(length, dtype=int)
    
    # Fast (Directional) Run Metrics
    trend_run_R = np.zeros(length, dtype=float)
    trend_duration_n = np.zeros(length, dtype=int)
    close_prev_r = np.zeros(length, dtype=float)
    t_start_fast: Union[int, None] = None # t_start for the fast (directional) run
    
    # Slow (Structural) Run Metrics
    structural_run_R = np.zeros(length, dtype=float)
    structural_duration_n = np.zeros(length, dtype=int)
    t_start_slow: Union[int, None] = None # t_start for the slow (structural) run
    
    # i is the loop index, starting from the first bar where t-3 exists (index 3).
    i = 3 

    while i < length:
        # 1. Determine the Structural Status (Ascending/Descending/None)
        is_ascending_holding = (l(i) >= l(i-3))
        is_descending_holding = (h(i) <= h(i-3))

        current_status = 0
        if is_ascending_holding:
            current_status = 1
        elif is_descending_holding:
            current_status = -1
        
        trend_status[i] = current_status
        
        # --- 2. Structural Run Start/Reset (t_start_slow determination) ---
        
        prev_status = trend_status[i-1]
        
        # Reset the structural run only if the structural trend sign changes or if it switches from 0 to +/-1
        if prev_status != current_status:
             # Set t_start_slow to the bar *before* the status changed (i-1)
             t_start_slow = i - 1

        # Calculate Structural Run R and n
        if t_start_slow is not None:
            structural_run_R[i] = c(i) - c(t_start_slow)
            structural_duration_n[i] = i - t_start_slow
            
        # --- 3. Directional Run Start/Reset (t_start_fast determination) ---
        
        current_close_r = c(i) - c(i-1)
        close_prev_r[i] = current_close_r
        
        current_R_sign = np.sign(current_close_r)
        
        # Check if a directional move occurred (c[i] != c[i-1])
        if current_R_sign != 0:
            
            prev_R_fast = trend_run_R[i-1]
            prev_R_sign_fast = np.sign(prev_R_fast)
            
            # Reset the directional run if:
            # A) t_start_fast hasn't been set yet (first move)
            # B) The direction has just reversed (current_R_sign != prev_R_sign_fast and prev_R_sign_fast != 0)
            
            if (t_start_fast is None) or (current_R_sign != prev_R_sign_fast and prev_R_sign_fast != 0):
                # Directional reversal detected or run initialization.
                t_start_fast = i - 1
            
        # Calculate Directional Run R and n
        if t_start_fast is not None:
            trend_run_R[i] = c(i) - c(t_start_fast)
            trend_duration_n[i] = i - t_start_fast
        
        i += 1
        
    data_frame['Close_Prev_R'] = close_prev_r
    data_frame['trend_status'] = trend_status
    data_frame['trend_run_R'] = trend_run_R
    data_frame['trend_duration_n'] = trend_duration_n
    data_frame['structural_run_R'] = structural_run_R
    data_frame['structural_duration_n'] = structural_duration_n
    
    return data_frame

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
        market_cap = ticker.info.get('marketCap')
        historical_data['relative_volume'] = historical_data['Volume'] / (market_cap / historical_data['Close'])
        historical_data = add_trend_info_column(historical_data)
        historical_data.to_csv(output_path)        

def load_market_data(symbol):
    module_dir = os.path.dirname(__file__)
    data_dir = os.path.join(module_dir, 'data')
    input_path = os.path.join(data_dir, f"{symbol}.csv")
    return pd.read_csv(input_path, parse_dates=True, date_format='%Y-%m-%d', index_col='Date')
