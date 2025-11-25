import pandas as pd
import yfinance as yf
import os
import re
import numpy as np
from typing import Callable, Union

from fourastro.market.indicators.classic import *

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
        add_astrological_longitude(historical_data)
        add_price_volume_strength(historical_data, 'Low')
        add_price_volume_strength(historical_data, 'High')
        add_price_volume_strength(historical_data, 'Close')
        add_average_true_range_percentage(historical_data, 14)
        add_bollinger_bands_width(historical_data, 'Low')
        add_bollinger_bands_width(historical_data, 'High')
        add_bollinger_bands_width(historical_data, 'Close')
        add_realized_volatility(historical_data, 'Low')
        add_realized_volatility(historical_data, 'High')
        add_realized_volatility(historical_data, 'Close')
        add_relative_volatility_index(historical_data, 14, 'High')
        add_relative_volatility_index(historical_data, 14, 'Low')
        add_relative_volume(ticker, historical_data)

        # 
        # add_bollinger_bands_width(historical_data, 20) # Required for G(t+1)
        # add_realized_volatility(historical_data) # Required for G(t+1)
        # add_relative_volatility_index(historical_data, 14)
        # add_fast_trend_run(historical_data) # input for breaking_gap
        # add_structural_direction(historical_data) # Input for slow_trend_run
        # add_slow_trend_run(historical_data) # input for breaking_gap
        # add_breaking_gap(historical_data) # input fast_swing_ratio
        # add_fast_swing_ratio(historical_data) #input for directional probabilities
        # add_slow_swing_ratio(historical_data) #input for directional probabilities
        # historical_data = add_directional_probabilities(historical_data)

        historical_data.to_csv(output_path)        

def load_market_data(symbol):
    module_dir = os.path.dirname(__file__)
    data_dir = os.path.join(module_dir, 'data')
    input_path = os.path.join(data_dir, f"{symbol}.csv")
    return pd.read_csv(input_path, parse_dates=True, date_format='%Y-%m-%d', index_col='Date')
