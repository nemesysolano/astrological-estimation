import pandas as pd
import yfinance as yf
import os
import re

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
        market_cap = ticker.info.get('marketCap')
        historical_data['relative_volume'] = historical_data['Volume'] / (market_cap / historical_data['Close'])
        historical_data.to_csv(output_path)        

def load_market_data(symbol):
    module_dir = os.path.dirname(__file__)
    data_dir = os.path.join(module_dir, 'data')
    input_path = os.path.join(data_dir, f"{symbol}.csv")
    return pd.read_csv(input_path, parse_dates=True, date_format='%Y-%m-%d', index_col='Date')
