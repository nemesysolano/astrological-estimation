from market import import_market_data
from analysis import analyze
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', type=str, help='Ticker symbol in NYSE')    
    parser.add_argument('price', type=str,  choices=['High', 'Low', 'Close'], help='The price to analyze', default='Close')    
    parser.add_argument('predictor', type=str, choices=['linear', 'gann'], help='The price to analyze', default='gann')
    args = parser.parse_args()
    ticker = args.ticker.upper()
    price = args.price
    predictor = args.predictor

    import_market_data(ticker)
    analyze(ticker, price, predictor)

