"""
Example demonstrating the Breeden-Litzenberger formula to extract
risk-neutral probability distributions from option prices.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.origin.distributions import calculate_distribution, calculate_moments

def generate_black_scholes_call_prices(S, K, r, T, sigma):
    """Generate call option prices using the Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_prices = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_prices

def main():
    # Parameters for the Black-Scholes model
    S0 = 100.0  # Current stock price
    r = 0.03    # Risk-free rate (3%)
    T = 0.25    # Time to expiration (3 months)
    sigma = 0.2  # Volatility (20%)
    
    # Generate a range of strike prices
    strikes = np.linspace(70, 130, 25)
    
    # Calculate call prices using Black-Scholes
    call_prices = generate_black_scholes_call_prices(S0, strikes, r, T, sigma)
    
    # Print the generated data
    print("Generated option data (first 5 rows):")
    for i in range(5):
        print(f"Strike: {strikes[i]:.2f}, Call Price: {call_prices[i]:.2f}")
    
    # Calculate the probability distribution using the Breeden-Litzenberger formula
    fine_strikes, density = calculate_distribution(
        strikes,
        call_prices,
        r,
        T,
        method="cubic_spline",
        num_points=1000
    )
    
    # Calculate distribution statistics
    moments = calculate_moments(fine_strikes, density)
    
    # Print the statistics
    print("\nDistribution Statistics:")
    print(f"Mean: {moments['mean']:.2f}")
    print(f"Standard Deviation: {moments['std_dev']:.2f}")
    print(f"Skewness: {moments['skewness']:.2f}")
    print(f"Kurtosis: {moments['kurtosis']:.2f}")
    
    # Plot the probability distribution
    plt.figure(figsize=(10, 6))
    plt.plot(fine_strikes, density, 'b-', linewidth=2)
    plt.fill_between(fine_strikes, density, alpha=0.3)
    
    # Add a vertical line at the mean
    plt.axvline(x=moments['mean'], color='r', linestyle='--', alpha=0.7, 
                label=f"Mean = {moments['mean']:.2f}")
    
    # Add a vertical line at the current stock price for reference
    plt.axvline(x=S0, color='g', linestyle=':', alpha=0.7,
                label=f"Current Price = {S0:.2f}")
    
    plt.title("Risk-Neutral Probability Distribution (Breeden-Litzenberger)")
    plt.xlabel("Price at Expiration")
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot instead of displaying it
    output_path = 'probability_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully to: {output_path}")

if __name__ == "__main__":
    main()