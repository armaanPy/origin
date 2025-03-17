import numpy as np

# Step 1: Define the calculation function with just the parts we've covered so far
def calculate_distribution_partial(
    strikes,
    call_prices,
    risk_free_rate,
    time_to_expiry,
    num_points=1000
):
    # Input validation
    if len(strikes) != len(call_prices):
        raise ValueError("Strikes and call prices must have the same length")
    
    if len(strikes) < 4:
        raise ValueError("Need at least 4 data points for reliable interpolation")
    
    # Simple monotonicity check (we'll implement the full version later)
    sorted_indices = np.argsort(strikes)
    sorted_strikes = np.array(strikes)[sorted_indices]
    sorted_call_prices = np.array(call_prices)[sorted_indices]
    
    for i in range(len(sorted_strikes) - 1):
        if sorted_call_prices[i] < sorted_call_prices[i+1]:
            raise ValueError("Call prices must decrease as strike prices increase")
    
    # Create a fine grid of strike prices
    fine_strikes = np.linspace(
        min(sorted_strikes), max(sorted_strikes), num_points
    )
    
    # Return what we've calculated so far
    return {
        "original_strikes": strikes,
        "original_call_prices": call_prices,
        "sorted_strikes": sorted_strikes,
        "sorted_call_prices": sorted_call_prices,
        "fine_strikes_first_5": fine_strikes[:5],
        "fine_strikes_last_5": fine_strikes[-5:],
        "num_fine_points": len(fine_strikes)
    }

# Step 2: Create test data
# A stock trading around $100 with various call option prices
test_strikes = np.array([110, 90, 100, 120, 80])  # Intentionally unsorted
test_call_prices = np.array([3.2, 12.0, 7.3, 1.1, 20.5])  # Corresponding prices
risk_free_rate = 0.03  # 3%
time_to_expiry = 0.25  # 3 months

# Step 3: Run the function and print results
try:
    result = calculate_distribution_partial(
        test_strikes,
        test_call_prices,
        risk_free_rate,
        time_to_expiry
    )
    
    print("Test successful! Results:")
    for key, value in result.items():
        print(f"{key}:")
        print(f"  {value}")
        print()
    
    # Verify sorting worked correctly
    print("Verification that sorting worked:")
    print("Original strikes:", test_strikes)
    print("Sorted strikes:", result["sorted_strikes"])
    print("Original call prices:", test_call_prices)
    print("Sorted call prices:", result["sorted_call_prices"])
    
except Exception as e:
    print(f"Test failed with error: {e}")