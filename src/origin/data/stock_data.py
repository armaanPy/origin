"""
Functions for fetching and analysing stock data from Yahoo Finance.
This module provides tools to retrieve historical prices and basic metrics.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

def fetch_stock_data(
    ticker: str,
    period: str = '1y',
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters:
    -----------
    ticker: str
        The stock symbol (e.g., 'AAPL', 'TSLA', 'GOOG')
    period: str, optional
        The period of historical data to fetch.
        ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    interval: str, optional
        The data time interval.
        ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

    Returns:
    --------
    pd.DataFrame
        DataFrame containing historical stock data.

    Example Usage:
    --------------
    data = fetch_stock_data('AAPL', period='1y', interval='1d')
    print(data)
    """
    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period, interval=interval)

        if hist_data.empty:
            raise ValueError(f"No historical data available for {ticker}")

        return hist_data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
    

def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Get general information about a stock.
    
    Parameters:
    -----------
    ticker : str
        The stock symbol
        
    Returns:
    --------
    Dict
        Dictionary containing stock information

    Example Usage:
    --------------
    info = get_stock_info('AAPL')
    print(info)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key metrics that are useful for options analysis
        key_metrics = {
            'symbol': ticker,
            'name': info.get('shortName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': info.get('regularMarketPrice', info.get('currentPrice', 'N/A')),
            'previous_close': info.get('previousClose', 'N/A'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'volume': info.get('volume', 'N/A'),
            'average_volume': info.get('averageVolume', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'eps': info.get('trailingEps', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'target_mean_price': info.get('targetMeanPrice', 'N/A'),
        }
        
        return key_metrics
    
    except Exception as e:
        print(f"Error fetching stock info for {ticker}: {e}")
        return {'symbol': ticker, 'error': str(e)}

def calculate_returns(
    price_data: pd.DataFrame,
    periods: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Calculate returns over various periods for stock price data.

    Parameters:
    -----------
    price_data: pd.DataFrame
        DataFrame with 'Close' prices.
    periods: Dict[str, int], optional
        Dictionary mapping period names to number of days.

    Returns:
    --------
    pd.DataFrame
        DataFrame with returns for each period.
    """
    if price_data.empty:
        return pd.DataFrame()

    if periods is None:
        periods = {
            'daily': 1,
            'weekly': 5,
            'monthly': 21,
            'quarterly': 63,
            'annual': 252
        }
    
    # Make a copy to avoid modifying the original data.
    returns_df = price_data[['Close']].copy()

    # Calculate returns for each period.
    for period_name, period_days in periods.items():
        if len(returns_df) > period_days:
            returns_df[f'{period_name}_return'] = returns_df['Close'].pct_change(period_days)
        
    # Calculate cumulative return.
    first_close = returns_df['Close'].iloc[0]
    last_close = returns_df['Close'].iloc[-1]
    total_return = (last_close / first_close) - 1
    
    # Calculate annualised volatility (standard deviation of daily returns * sqrt(252)).
    daily_returns = returns_df['Close'].pct_change().dropna()
    annual_volatility = daily_returns.std() * (252 ** 0.5)
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Volatility: {annual_volatility:.2%}")
    
    return returns_df