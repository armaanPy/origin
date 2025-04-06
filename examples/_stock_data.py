"""
Example demonstrating how to use the stock data functions.
"""

import matplotlib.pyplot as plt
import pandas as pd
from src.origin.data.stock_data import fetch_stock_data, get_stock_info, calculate_returns

def main():
    # Define the ticker to analyse
    ticker = "AAPL"
    
    # Get stock information
    print(f"Fetching information for {ticker}...")
    stock_info = get_stock_info(ticker)
    
    # Print key stock information
    print("\nStock Information:")
    for key, value in stock_info.items():
        print(f"{key}: {value}")
    
    # Fetch historical stock data
    print(f"\nFetching historical data for {ticker}...")
    hist_data = fetch_stock_data(ticker, period="1y")
    
    if hist_data.empty:
        print(f"Could not retrieve historical data for {ticker}")
        return
    
    # Display the first few rows of historical data
    print("\nHistorical Data Preview:")
    print(hist_data.head())
    
    # Calculate returns
    print("\nCalculating returns...")
    returns_df = calculate_returns(hist_data)
    
    # Plot the stock price
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    hist_data['Close'].plot(title=f"{ticker} Stock Price (Past Year)")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    returns_df['daily_return'].plot(title=f"{ticker} Daily Returns")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure instead of displaying it (for containers without GUI)
    plt.savefig('stock_analysis.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'stock_analysis.png'")

if __name__ == "__main__":
    main()