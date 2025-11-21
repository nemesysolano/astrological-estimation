from fourastro import market
import numpy as np
import sys
import os

def price_volume_oscillator(historical_data, price, t):
    index = historical_data.index
    # Calculates Y(t) as defined in README.md
    if t == 0:
        return 0.0

    p_t = historical_data[price][index[t]]
    p_t_minus_1 = historical_data[price][index[t-1]]

    v_t = historical_data['Volume'][index[t]]
    v_t_minus_1 = historical_data['Volume'][index[t-1]]

    price_sum = p_t + p_t_minus_1
    if price_sum == 0 or v_t_minus_1 == 0:
        return 0.0

    price_component = (p_t - p_t_minus_1) / price_sum
    # README says min(1, v_t / v_t-1)
    volume_component = min(1, v_t / v_t_minus_1)
    
    return price_component * volume_component

if __name__ == "__main__":
    ticker = sys.argv[1]
    market.import_market_data(ticker)
    historical_data = market.load_market_data(ticker)
    index = historical_data.index
    close = lambda t: historical_data.loc[index[t], 'Close']
    low = lambda t: historical_data.loc[index[t], 'Low']
    high = lambda t: historical_data.loc[index[t], 'High']
    p_up = lambda t: historical_data.loc[index[t], 'p_up']
    p_down = lambda t: historical_data.loc[index[t], 'p_down']
    matches = []
    up_count = 0
    down_count = 0
    time_delta = 1
    gauge_up = low
    gauge_down = high
    
    y_t_close = lambda t: price_volume_oscillator(historical_data, 'Close', t)
    y_t_high = lambda t: price_volume_oscillator(historical_data, 'High', t )
    y_t_low = lambda t: price_volume_oscillator(historical_data, 'Low', t)

    for t in range(1, len(index)-time_delta):
            sign = 0
            
              # UP CONDITION: (Probability is UP OR Momentum is Positive) AND (Higher Low)
            if (p_up(t) > p_down(t) or y_t_low(t) > y_t_low(t-1)) and gauge_up(t+time_delta) > gauge_up(t):
                sign = 1
                up_count += 1
            
            # DOWN CONDITION: (Probability is DOWN OR Momentum is Negative) AND (Lower High)
            elif (p_up(t) < p_down(t) or y_t_high(t) < y_t_high(t-1)) and gauge_down(t+time_delta) < gauge_down(t):
                sign = -1
                down_count += 1
            
            matches.append(sign)

    test_results_dir = os.path.join(os.getcwd(), 'test_results')
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
    test_results_file = os.path.join(test_results_dir, 'directional_probabilities.md')       
    open_mode = 'a' if os.path.exists(test_results_file) else 'w'

    with open(test_results_file, open_mode) as f:
        if open_mode == 'w':
            print("| ticker | up count | down count | matches | matches %|", file=f)
            print("|:---|---:|---:|---:|---:|", file=f)
        match_count = np.count_nonzero(matches)        
        if len(matches) > 0:
            maches_pct = match_count / len(matches) * 100
        else:
            maches_pct = 0
        print(f"| {ticker} | {up_count} | {down_count} | {match_count} | {maches_pct:.2f} %|", file=f)
