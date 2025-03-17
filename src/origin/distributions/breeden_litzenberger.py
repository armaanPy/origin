"""
Breeden-Litzenberger (1978) formula implementation for extracting risk-neutral probability
distributions from option prices.

This theorem states that the second derivative of the call price with respect to the strike
price equals the discounted risk-neutral probability density function.
"""

import numpy as np
import scipy.interpolate as interp
from typing import Tuple, Optional, Callable, Union, Dict

def calculate_distribution(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    risk_free_rate: float,
    time_to_expiry: float,
    method: str = "cubic_spline",
    num_points: int = 1000,
    smoothing_factor: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the risk-neutral probability distribution using the Breeden-Litzenberger formula.

    The formula states that the second derivative of the call price with respect to the strike
    price equals the discounted risk-neutral probability density:

    Parameters
    ----------
    strikes: np.ndarray
        Array of strike prices.
    call_prices: np.ndarray
        Array of call option prices corresponding to the strikes.
    risk_free_rate: float
        The risk-free interest rate (as a decimal, e.g., 0.03 for 3%)
    time_to_expiry: float
        The time to expiration of the options in years.
    method: str, optional
        The interpolation method to use for the call prices.
        Options: "cubic_spline" (default), "linear", "quadratic", "cubic"
    num_points: int, optional
        The number of points to use for the fine grid of the distribution.
    smoothing_factor: float, optional
        The smoothing factor for smoothing_spline method.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (fine_strikes, probability_density)
        Two arrays:
        - The first array contains the strike prices.
        - The second array contains the risk-neutral probability density function.
    """
    # Input validation to ensure the data meets the necessary mathematical requirements for the BL formula.
    if len(strikes) != len(call_prices):
        raise ValueError("Strikes and call prices must have the same length.")

    if len(strikes) < 4:
        raise ValueError("At least 4 strikes are needed for reliable interpolation.")

    if not _is_monotonically_decreasing(strikes, call_prices):
        raise ValueError("Call prices must decrease as strike prices increase.")

    # Prepare data for processing.
    # Sort data by strike prices.
    sorted_indices = np.argsort(strikes)
    sorted_strikes = np.array(strikes)[sorted_indices]
    sorted_call_prices = np.array(call_prices)[sorted_indices]

    # Create a fine grid of strike prices for the distribution.
    fine_strikes = np.linspace(
        min(sorted_strikes),
        max(sorted_strikes),
        num_points
    )

    # Interpolation method selection.
    if method == "cubic_spline":
        second_derivatives = _cubic_spline_interpolation(
            sorted_strikes,
            sorted_call_prices,
            fine_strikes,
        )
    elif method == "smoothing_spline":
        second_derivatives = _smoothing_spline_interpolation(
            sorted_strikes,
            sorted_call_prices,
            fine_strikes,
            smoothing_factor
        )
    elif method == "polynomial":
        second_derivatives = _polynomial_interpolation(
            sorted_strikes,
            sorted_call_prices,
            fine_strikes
        )
    else:
        raise ValueError(f"Invalid interpolation method: {method}. Must be one of: 'cubic_spline', 'smoothing_spline', or 'polynomial'")
        
    # Apply the Breeden-Litzenberger formula to get the probability density.
    discount_factor = np.exp(risk_free_rate * time_to_expiry)
    probability_density = discount_factor * second_derivatives

    # Post-process the probability density to ensure non-negativity and integration to 1.
    probability_density = np.maximum(0, probability_density)
    area = np.trapezoid(probability_density, fine_strikes)
    if area > 0:
        probability_density = probability_density / area
        return fine_strikes, probability_density
    else:
        raise ValueError("The area under the probability density is not positive. This is likely due to the call prices being too low or the risk-free rate being too high.")
    
def _is_monotonically_decreasing(
    x: np.ndarray, y: np.ndarray
) -> bool:
    """
    Check if y values monotonically decrease as x values increase.

    Parameters:
    -----------
    x: np.ndarray
        Array of x values (strike prices).
    y: np.ndarray
        Array of y values (call prices).

    Returns:
    --------
    bool: True if y values monotonically decrease with x, False otherwise.
    """

    # Sort by x values.
    sorted_indices = np.argsort(x)
    sorted_y = np.array(y)[sorted_indices]

    # Check if differences are non-positive (decreasing or flat).
    return np.all(np.diff(sorted_y) <= 0)

def _cubic_spline_interpolation(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    fine_strikes: np.ndarray
) -> np.ndarray:
    """
    Use cubic spline interpolation to calculate the second derivative.
    
    Parameters:
    -----------
    strikes : np.ndarray
        Sorted array of strike prices
    call_prices : np.ndarray
        Corresponding array of call prices
    fine_strikes : np.ndarray
        Fine grid of strike prices for interpolation
        
    Returns:
    --------
    np.ndarray
        Second derivatives at the fine grid points
    """
    cs = interp.CubicSpline(strikes, call_prices)
    return cs.derivative(2)(fine_strikes)

def _smoothing_spline_interpolation(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    fine_strikes: np.ndarray,
    smoothing_factor: Optional[float] = None
) -> np.ndarray:
    """
    Use smoothing spline interpolation to calculate the second derivative.
    
    Parameters:
    -----------
    strikes : np.ndarray
        Sorted array of strike prices
    call_prices : np.ndarray
        Corresponding array of call prices
    fine_strikes : np.ndarray
        Fine grid of strike prices for interpolation
    smoothing_factor : float, optional
        Smoothing factor (s parameter in UnivariateSpline)
        If None, defaults to len(strikes)
        
    Returns:
    --------
    np.ndarray
        Second derivatives at the fine grid points
    """
    # Default smoothing factor if not provided.
    if smoothing_factor is None:
        smoothing_factor = len(strikes)
        
    spl = interp.UnivariateSpline(
        strikes, call_prices, k=3, s=smoothing_factor
    )
    return spl.derivative(2)(fine_strikes)

def _polynomial_interpolation(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    fine_strikes: np.ndarray
) -> np.ndarray:
    """
    Use polynomial interpolation to calculate the second derivative.
    
    Parameters:
    -----------
    strikes : np.ndarray
        Sorted array of strike prices
    call_prices : np.ndarray
        Corresponding array of call prices
    fine_strikes : np.ndarray
        Fine grid of strike prices for interpolation
        
    Returns:
    --------
    np.ndarray
        Second derivatives at the fine grid points
    """
    # Use a degree that balances fit and stability.
    degree = min(5, len(strikes) - 1)
    poly_coeffs = np.polyfit(strikes, call_prices, degree)
    
    # Calculate the second derivative coefficients.
    second_deriv_coeffs = np.polyder(poly_coeffs, 2)
    
    # Evaluate the second derivative polynomial at the fine grid points.
    return np.polyval(second_deriv_coeffs, fine_strikes)

# Calculate the moments of the distribution.
def calculate_moments(
    strikes: np.ndarray,
    density: np.ndarray,
    max_moment: int = 4
) -> Dict[str, float]:
    """
    Calculate statistical moments of the probability distribution.
    
    Parameters:
    -----------
    strikes : np.ndarray
        Array of strike prices (fine grid)
    density : np.ndarray
        Array of probability densities
    max_moment : int, optional
        Maximum moment to calculate
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with moments (mean, variance, skewness, kurtosis)
    """
    moments = {}
    
    # Calculate the mean (1st raw moment).
    mean = np.trapz(strikes * density, strikes)
    moments["mean"] = mean
    
    if max_moment >= 2:
        # Calculate the 2nd central moment (variance).
        var_integrand = ((strikes - mean) ** 2) * density
        variance = np.trapz(var_integrand, strikes)
        moments["variance"] = variance
        moments["std_dev"] = np.sqrt(variance)
    
    if max_moment >= 3:
        # Calculate the 3rd central moment (for skewness)
        skew_integrand = ((strikes - mean) ** 3) * density
        third_moment = np.trapz(skew_integrand, strikes)
        # Skewness = 3rd central moment / std_dev^3
        if variance > 0:
            moments["skewness"] = third_moment / (np.sqrt(variance) ** 3)
        else:
            moments["skewness"] = 0
    
    if max_moment >= 4:
        # Calculate the 4th central moment (for kurtosis).
        kurt_integrand = ((strikes - mean) ** 4) * density
        fourth_moment = np.trapz(kurt_integrand, strikes)
        # Kurtosis = 4th central moment / variance^2
        if variance > 0:
            moments["kurtosis"] = fourth_moment / (variance ** 2)
        else:
            moments["kurtosis"] = 0
    
    return moments