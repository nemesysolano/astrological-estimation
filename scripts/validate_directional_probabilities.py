from fourastro import market
import numpy as np
import sys
import os

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

    # ... inside the loop
    for t in range(len(index)-time_delta):
            sign = 0
            # Check if we predicted UP and the High price went to a new high
            if p_up(t) > p_down(t) and gauge_up(t+time_delta) > gauge_up(t):
                sign = 1
                up_count += 1
            # Check if we predicted DOWN and the Low price went to a new low
            elif p_up(t) < p_down(t) and gauge_down(t+time_delta) < gauge_down(t):
                sign = -1
                down_count += 1
            matches.append(sign)
        # ...

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
