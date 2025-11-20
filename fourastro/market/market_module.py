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
        add_fast_trend_run(historical_data)
        add_structural_direction(historical_data)
        add_slow_trend_run(historical_data)

        historical_data.to_csv(output_path)        

def load_market_data(symbol):
    module_dir = os.path.dirname(__file__)
    data_dir = os.path.join(module_dir, 'data')
    input_path = os.path.join(data_dir, f"{symbol}.csv")
    return pd.read_csv(input_path, parse_dates=True, date_format='%Y-%m-%d', index_col='Date')
