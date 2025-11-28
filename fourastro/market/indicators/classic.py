
import pandas as pd
import numpy as np


def add_price_volume_strength(historical_data, price):
    price_series = historical_data[price]
    volume_series = historical_data['Volume']

    p_t = price_series
    p_t_minus_1 = price_series.shift(1)
    v_t = volume_series
    v_t_minus_1 = volume_series.shift(1)

    price_change_ratio = (p_t - p_t_minus_1) / (p_t + p_t_minus_1)

    volume_ratio = v_t / v_t_minus_1
    capped_volume_ratio = np.minimum(1, volume_ratio)
    y_t = price_change_ratio * capped_volume_ratio    

    historical_data[f'Y_{price}'] =  y_t
    historical_data.dropna(inplace=True)

def add_average_true_range_percentage(historical_data, period):
    high = historical_data['High']
    low = historical_data['Low']
    close = historical_data['Close']

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Average True Range (ATR) using an Exponential Moving Average
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    # Calculate ATRP and add it to the DataFrame
    historical_data[f'ATRP'] = atr / close
    historical_data.dropna(inplace=True)    

def add_bollinger_bands_width(historical_data, price):
    period = 20
    close_prices = historical_data[price]

    # Calculate the Middle Band (Simple Moving Average)
    middle_band = close_prices.rolling(window=period).mean()

    # Calculate the Standard Deviation
    std_dev = close_prices.rolling(window=period).std()

    # Calculate Upper and Lower Bands (2 standard deviations is common)
    upper_band = middle_band + (2 * std_dev)
    lower_band = middle_band - (2 * std_dev)

    historical_data[f"BBW_{price}"] = (upper_band - lower_band) / middle_band
    historical_data.dropna(inplace=True)    

def add_realized_volatility(historical_data, price):
    window = 5
    log_returns = np.log(historical_data[price] / historical_data[price].shift(1))
    
    # The Realized Volatility is the square root of the sum of squared log returns.
    realized_volatility = log_returns.rolling(window=window).std() * np.sqrt(window)
    historical_data[f"RVO_{price}"] = realized_volatility
    historical_data.dropna(inplace=True)    

def add_relative_volatility_index(historical_data, period, price):
    close_prices = historical_data[price]
    std_dev = close_prices.rolling(window=period).std()

    # Calculate price changes
    price_change = close_prices.diff()

    # Calculate Upward and Downward Volatility
    up_vol = np.where(price_change > 0, std_dev, 0)
    down_vol = np.where(price_change < 0, std_dev, 0)

    # Calculate Exponential Moving Averages of Up and Down Volatility
    avg_up_vol = pd.Series(up_vol, index=historical_data.index).ewm(span=period, adjust=False).mean()
    avg_down_vol = pd.Series(down_vol, index=historical_data.index).ewm(span=period, adjust=False).mean()

    # Calculate RVI
    rvi = 100 * avg_up_vol / (avg_up_vol + avg_down_vol)
    historical_data[f"RVI_{price}"] = rvi
    historical_data.dropna(inplace=True)

def add_relative_volume(ticker, historical_data):
    market_cap = ticker.info.get('marketCap')
    historical_data['RV'] = historical_data['Volume'] / (market_cap / historical_data['Close'])    
    historical_data.dropna(inplace=True)        