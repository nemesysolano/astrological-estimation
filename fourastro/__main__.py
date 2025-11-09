from astro import initialize_astro_data
from market import import_market_data
from analysis import forecast
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', type=str, help='Ticker symbol in NYSE')    
    args = parser.parse_args()
    ticker = args.ticker.upper()

    initialize_astro_data()
    import_market_data(ticker)
    forecast(ticker)

