"""Finance Functions v.0.1

A binomial option tree pricing model function I have created during my time at the University of Toronto based
on material from  Derivative Markets and International Financial Management. Constantly
being updated to include more course material and general improvements. Definitely need to take into account data-scraping with
yfinance and also create a Black-Scholes function to add to here.

Copyright (c) 2024 Lunes Maibeche

"""

import numpy as np
import time
import cProfile
import numba


# Binomial Tree Algorithm Outline
# Arguments will be stock ticker/spot price, exercise, number of steps, put or call, american or european,
# time to maturity, volatility , rf and dividend curve
# 1. First thing would be having to set p, u and d
# 2. Generate end nodes and get option value there
# 3. Collapse back to the present
# Note it's essentially just a recurisve binary tree

# TO-DO LIST:
#   1. Create an alternative put branch for european no dividend [X] 
#   2. Create an American early exercise checker [X]
#   3. Improve that runtime! []
#   4. Implement dividends branch []
#   5. Further refine for stock indexes, currencies, futures [X]
#   6. Create doctests [X]
#   9. Check edge cases and make more user-friendly with prompts []
#   10. Implement yfinance stock price checker, rf checker, volatility checker and dividend checker []
# 1.399594 secs with numba vs 1.387312 seconds without but on average numba looks faster - might need to look into
# fixing the print answer problem

def binomial_tree(spot_price: float, strike_price: float, time_to_maturity: float, r: float,
                  volatility: float, steps: int) -> dict:
    """Return the possible prices of an option on a given stock given the rf rate, volatility,
    time to maturity and option type. The price estimate is found by setting up a binomial tree with a user-set number
    of steps.
    Note: For the time being, only addresses no-dividend options on stock
    >>> binomial_tree(65.5, 65, 10/12, 0.048, 0.325, 10000)
        {'European' : {'Call': 9.165316766182787, 'Put' : 6.116630311125335}, 'American': {'Call': 9.165316766182787,
        'Put' : 6.116630311125335}}
    >>> binomial_tree(65.5, 68, 12/12, 0.048, 0.325, 3)
        {'European' : {'Call': 8.75884096486369, 'Put' : 8.071938486124544}, 'American': {'Call': 8.75884096486369,
        'Put' : 8.071938486124544}}
    """
    type = input('Please select an asset type from: stock, index, currency, future.').lower()

    # Precalculate essential variables
    print('Creating starting variables...')
    result = {'European': {'Call': 0.0, 'Put': 0.0}, 'American': {'Call': 0.0, 'Put': 0.0}}
    dt = time_to_maturity / steps  # Intervals of time between steps
    u = np.exp(volatility * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    dsc = np.exp(-r * dt)  # Risk-free continuously compounding discounting by one dt

    pu = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability of going up for index options
    function_start = time.perf_counter()
    start_time = time.perf_counter()
    pud = dsc * pu
    pd = 1 - pu  # Risk-neutral probability of going down for stock options
    pdd = dsc * pd

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Variables runtime: {elapsed_time:.6f} seconds")  # 0.00029 seconds to execute

    print('Generating spot prices in tree...')
    start_time = time.perf_counter()
    tree_start = time.perf_counter()

    spot_prices = np.zeros((steps + 1, steps + 1))
    for N in np.arange(steps, -1, -1):
        spot_prices[:N + 1, N] = spot_price * u ** np.arange(N, -N - 1, -2)
    non_zero_mask = spot_prices != 0

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Spot price generation runtime: {elapsed_time:.6f} seconds")  # 0.694954 seconds to execute

    print('Generating exercise option prices in tree...')
    start_time = time.perf_counter()

    call_prices = np.zeros((steps + 1, steps + 1))
    put_prices = np.zeros((steps + 1, steps + 1))
    call_prices[non_zero_mask] = np.maximum(spot_prices[non_zero_mask] - strike_price, 0)
    put_prices[non_zero_mask] = np.maximum(strike_price - spot_prices[non_zero_mask], 0)
    american_call = np.copy(call_prices)
    american_put = np.copy(put_prices)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Exercise option price generation runtime: {elapsed_time:.6f} seconds")  # 1.422079 seconds to execute

    print('Generating European option prices in tree...')
    start_time = time.perf_counter()

    for N in np.arange(steps - 1, -1, -1):
        call_prices[:N + 1, N] = (pud * call_prices[:N + 1, N + 1] + pdd * call_prices[1:N + 2, N + 1])
        put_prices[:N + 1, N] = (pud * put_prices[:N + 1, N + 1] + pdd * put_prices[1:N + 2, N + 1])
    result['European']['Call'] = float(call_prices[0][0])
    result['European']['Put'] = float(put_prices[0][0])

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"European option price generation runtime: {elapsed_time:.6f} seconds")  # 0.817160 seconds to execute

    # This step takes way too long, there must be a way to integrate it into the european step
    print('Generating American option prices in tree...')
    start_time = time.perf_counter()

    # The loop is the problematic part, but how else can I do this...
    for N in np.arange(steps - 1, -1, -1):
        recursive_call = pud * american_call[:N + 1, N + 1] + pdd * american_call[1:N + 2, N + 1]
        recursive_put = pud * american_put[:N + 1, N + 1] + pdd * american_put[1:N + 2, N + 1]

        early_exercise_call = spot_prices[:N + 1, N] - strike_price
        early_exercise_put = strike_price - spot_prices[:N + 1, N]

        american_call[:N + 1, N] = np.maximum(recursive_call, early_exercise_call)
        american_put[:N + 1, N] = np.maximum(recursive_put, early_exercise_put)

    result['American']['Call'] = float(american_call[0, 0])
    result['American']['Put'] = float(american_put[0, 0])

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"American option price generation runtime: {elapsed_time:.6f} seconds")  # 0.995407 seconds to execute

    end_time = time.perf_counter()
    tree_time_elapse = end_time - tree_start
    function_time_elapse = end_time - function_start
    print(f"Tree generation runtime: {tree_time_elapse:.6f} seconds")  # 3.929894 seconds to execute total
    print(f"Function runtime: {function_time_elapse:.6f} seconds")  # 3.929934 seconds to execute total
    return result


# If you want to implement a specifier, could be done pretty easily
# option_type = option_info.split()[1]
# option_country = option_info.split()[0]

# NOTE - I hate compromising a possibly more efficient way to calculate european options only
# because I will need American as well - hold onto this
# if 'option_type = 'European':
# end_layer_prices = np.array([spot_price * u ** (steps -2 * i) for i in range(steps+1)])
# call_prices = np.maximum(end_layer_prices - strike_price, 0.0)
# put_prices = np.maximum(strike_price - end_layer_prices, 0.0)

# Collapse everything back to find call price for european option (USE FOR LOOP INSTEAD OF WHILE)
# while len(call_prices) > 1:
#    call_prices = np.maximum(0, dsc * (pd * call_prices[1:] + pu * call_prices[:-1]))
#    put_prices = np.maximum(0, dsc * (pd * put_prices[1:] + pu * put_prices[:-1]))

# result['European']['Call'] = call_prices[0]
# result['European']['Put'] = put_prices[0]

# start_time = time.time()
# cumtime = 0
#     for i in range(10000):
#         start_time = time.perf_counter()
#
#         end_time = time.perf_counter()
#         elapsed_time = end_time - start_time
#         cumtime += elapsed_time
# average_time = cumtime/10000
# print(f"Average Runtime: {average_time:.6f} seconds")


# binomial_tree(65.5, 65, 10/12, 0.048, 0.325, 5000)
