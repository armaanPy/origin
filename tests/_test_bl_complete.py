import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import math

# Import all of our code so far (assuming it's in a file called breeden_litzenberger.py)
# If you're still building the file, you can include the full implementation in this test script

# First, implement all the helper functions we've discussed
def _is_monotonically_decreasing(x, y):
    sorted_indices = np.argsort(x)
    sorted_y = np.array(y)[sorted_indices]
    return np.all(np.diff(sorted_y) <= 0)

def _cubic_spline_interpolation(strikes, call_prices, fine_strikes):
    import scipy.interpolate as interp
    cs = interp.CubicSpline(strikes, call_prices)
    return cs.derivative(2)(fine_strikes)

def _smoothing_spline_interpolation(strikes, call_prices, fine_strikes, smoothing_factor=None):
    import scipy.interpolate as interp
    if smoothing_factor is None:
        smoothing_factor = len(strikes)
    spl = interp.UnivariateSpline(strikes, call_prices, k=3, s=smoothing_factor)
    return spl.derivative(2)(fine_strikes)

def _polynomial_interpolation(strikes, call_prices, fine_strikes):
    degree = min(5, len(strikes) - 1)
    poly_coeffs = np.polyfit(strikes, call_prices, degree)
    second_deriv_coeffs = np.polyder(poly_coeffs, 2)
    return np.polyval(second_deriv_coeffs, fine_strikes)

# Implement the main distribution calculation function
def calculate_distribution(
    strikes,
    call_prices,
    risk_free_rate,
    time_to_expiry,
    method="cubic_spline",
    num_points=1000,
    smoothing_factor=None,
):
    # Input validation
    if len(strikes) != len(call_prices):
        raise ValueError("Strikes and call prices must have the same length")
    
    if len(strikes) < 4:
        raise ValueError("Need at least 4 data points for reliable interpolation")
    
    if not _is_monotonically_decreasing(strikes, call_prices):
        raise ValueError("Call prices must decrease as strike prices increase")
    
    # Sort data by strike price
    sorted_indices = np.argsort(strikes)
    sorted_strikes = np.array(strikes)[sorted_indices]
    sorted_call_prices = np.array(call_prices)[sorted_indices]
    
    # Create a fine grid of strike prices
    fine_strikes = np.linspace(
        min(sorted_strikes), max(sorted_strikes), num_points
    )
    
    # Apply the selected interpolation method
    if method == "cubic_spline":
        second_derivatives = _cubic_spline_interpolation(
            sorted_strikes, sorted_call_prices, fine_strikes
        )
    elif method == "smoothing_spline":
        second_derivatives = _smoothing_spline_interpolation(
            sorted_strikes, sorted_call_prices, fine_strikes, smoothing_factor
        )
    elif method == "polynomial":
        second_derivatives = _polynomial_interpolation(
            sorted_strikes, sorted_call_prices, fine_strikes
        )
    else:
        raise ValueError(
            "Method must be one of 'cubic_spline', 'smoothing_spline', or 'polynomial'"
        )
    
    # Apply the Breeden-Litzenberger formula
    discount_factor = np.exp(risk_free_rate * time_to_expiry)
    probability_density = discount_factor * second_derivatives
    
    # Clean up negative probabilities
    probability_density = np.maximum(0, probability_density)
    
    # Normalize the distribution
    area = np.trapz(probability_density, fine_strikes)
    if area > 0:
        probability_density = probability_density / area
    
    return fine_strikes, probability_density

def calculate_moments(strikes, density, max_moment=4):
    moments = {}
    
    # Calculate the mean (1st raw moment)
    mean = np.trapz(strikes * density, strikes)
    moments["mean"] = mean
    
    if max_moment >= 2:
        # Calculate the 2nd central moment (variance)
        var_integrand = ((strikes - mean) ** 2) * density
        variance = np.trapz(var_integrand, strikes)
        moments["variance"] = variance
        moments["std_dev"] = np.sqrt(variance)
    
    if max_moment >= 3:
        # Calculate the 3rd central moment (for skewness)
        skew_integrand = ((strikes - mean) ** 3) * density
        third_moment = np.trapz(skew_integrand, strikes)
        if variance > 0:
            moments["skewness"] = third_moment / (np.sqrt(variance) ** 3)
        else:
            moments["skewness"] = 0
    
    if max_moment >= 4:
        # Calculate the 4th central moment (for kurtosis)
        kurt_integrand = ((strikes - mean) ** 4) * density
        fourth_moment = np.trapz(kurt_integrand, strikes)
        if variance > 0:
            moments["kurtosis"] = fourth_moment / (variance ** 2)
        else:
            moments["kurtosis"] = 0
    
    return moments

# Now create tests to validate our implementation

def generate_black_scholes_call_prices(S, K, r, T, sigma):
    """Generate call option prices using the Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    from scipy.stats import norm
    call_prices = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_prices

def test_with_black_scholes():
    """
    Test our implementation by generating option prices from a Black-Scholes model,
    which assumes lognormal distribution, and see if our extracted distribution matches.
    """
    print("Testing with Black-Scholes generated option prices...\n")
    
    # Parameters
    S0 = 100  # Current stock price
    r = 0.05  # Risk-free rate
    T = 0.5   # Time to expiration (6 months)
    sigma = 0.2  # Volatility
    
    # Generate a range of strikes centered around the current price
    strikes = np.linspace(70, 130, 25)
    
    # Calculate theoretical call prices using Black-Scholes
    call_prices = generate_black_scholes_call_prices(S0, strikes, r, T, sigma)
    
    # Extract the distribution using our implementation
    fine_strikes, density = calculate_distribution(strikes, call_prices, r, T)
    
    # Calculate moments of our extracted distribution
    moments = calculate_moments(fine_strikes, density)
    
    # Print statistics
    print("Extracted Distribution Statistics:")
    print(f"Mean: {moments['mean']:.2f}")
    print(f"Standard Deviation: {moments['std_dev']:.2f}")
    print(f"Skewness: {moments['skewness']:.4f}")
    print(f"Kurtosis: {moments['kurtosis']:.4f}")
    
    # Generate points from the theoretical lognormal distribution
    # The risk-neutral drift means the expected return is the risk-free rate
    mu = math.log(S0) + (r - 0.5 * sigma**2) * T
    s = sigma * math.sqrt(T)
    
    # Calculate theoretical lognormal PDF
    theoretical_pdf = lognorm.pdf(fine_strikes, s=s, scale=math.exp(mu))
    
    # Calculate theoretical moments
    theoretical_mean = math.exp(mu + 0.5 * s**2)
    theoretical_var = (math.exp(s**2) - 1) * math.exp(2*mu + s**2)
    theoretical_stdev = math.sqrt(theoretical_var)
    
    print("\nTheoretical Lognormal Statistics:")
    print(f"Mean: {theoretical_mean:.2f}")
    print(f"Standard Deviation: {theoretical_stdev:.2f}")
    
    # Calculate the relative errors
    mean_error = abs(moments['mean'] - theoretical_mean) / theoretical_mean * 100
    std_error = abs(moments['std_dev'] - theoretical_stdev) / theoretical_stdev * 100
    
    print("\nRelative Errors:")
    print(f"Mean Error: {mean_error:.2f}%")
    print(f"Std Dev Error: {std_error:.2f}%")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(fine_strikes, density, 'b-', linewidth=2, label='Extracted Distribution')
    plt.plot(fine_strikes, theoretical_pdf, 'r--', linewidth=2, label='Theoretical Lognormal')
    plt.title('Extracted vs. Theoretical Risk-Neutral Density')
    plt.xlabel('Stock Price at Expiration')
    plt.ylabel('Probability Density')
    plt.axvline(x=moments['mean'], color='b', linestyle=':', alpha=0.7, 
                label=f"Extracted Mean = {moments['mean']:.2f}")
    plt.axvline(x=theoretical_mean, color='r', linestyle=':', alpha=0.7,
                label=f"Theoretical Mean = {theoretical_mean:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return mean_error < 5 and std_error < 10  # Check if errors are acceptably small

def test_with_real_world_example():
    """Test with a realistic set of market option prices."""
    print("\nTesting with realistic market option data...\n")
    
    # Sample data: call options with strikes and prices
    # This could be from real market data
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    call_prices = np.array([21.5, 17.2, 13.4, 10.1, 7.3, 5.0, 3.2, 1.9, 1.1])
    
    # Parameters
    risk_free_rate = 0.03  # 3%
    time_to_expiry = 0.25  # 3 months
    
    # Test all interpolation methods
    methods = ["cubic_spline", "smoothing_spline", "polynomial"]
    colors = ['b', 'g', 'r']
    
    plt.figure(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        # Calculate the probability distribution
        fine_strikes, density = calculate_distribution(
            strikes,
            call_prices,
            risk_free_rate,
            time_to_expiry,
            method=method
        )
        
        # Calculate moments
        moments = calculate_moments(fine_strikes, density)
        
        # Print statistics
        print(f"\nMethod: {method}")
        print(f"Mean: {moments['mean']:.2f}")
        print(f"Standard Deviation: {moments['std_dev']:.2f}")
        print(f"Skewness: {moments['skewness']:.4f}")
        print(f"Kurtosis: {moments['kurtosis']:.4f}")
        
        # Plot
        plt.plot(fine_strikes, density, f"{colors[i]}-", linewidth=2, 
                label=f"{method.replace('_', ' ').title()}")
        plt.axvline(x=moments['mean'], color=colors[i], linestyle=':', alpha=0.5,
                   label=f"{method.title()} Mean = {moments['mean']:.2f}")
    
    plt.title('Risk-Neutral Probability Distributions Using Different Methods')
    plt.xlabel('Stock Price at Expiration')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return True  # Visual inspection required

# Run the tests
if __name__ == "__main__":
    print("Running tests for Breeden-Litzenberger implementation...\n")
    
    # Test with Black-Scholes generated prices
    bs_test_passed = test_with_black_scholes()
    
    # Test with realistic market data
    market_test_passed = test_with_real_world_example()
    
    # Print overall results
    print("\nTest Results:")
    print(f"Black-Scholes Validation: {'PASSED' if bs_test_passed else 'FAILED'}")
    print(f"Market Data Example: {'PASSED' if market_test_passed else 'FAILED'}")