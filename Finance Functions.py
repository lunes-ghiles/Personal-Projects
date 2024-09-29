"""Finance Functions v.0.1

A collection of finance related functions I have created during my time at the University of Toronto based
on material from Introduction to Finance I&II, Derivative Markets, International Financial Management. Constantly
being updated to include more course material and general improvements.

Copyright (c) 2024 Lunes Maibeche

"""

import math
import datetime as dt
import yfinance as yf

# Constants used
DATE_SEP = '-'
COL_SEP = ','
COL_DATE = 0
COL_OPEN = 1
COL_HIGH = 2
COL_LOW = 3
COL_CLOSE = 4
COL_ADJ_CLOSE = 5
start = dt.datetime(2019, 1, 1)
end = dt.datetime.now()


def rf_estimate() -> float:
    """Use the average 10-year zero coupon US treasury bond rate since the last FOMC announcement till today as current
    risk-free rate estimate. Unfortunately, I have not figured out how to automatically extract the most recent
    FOMC announcement from the FRB website, so for now, the function requires the most recent date as a given.
    """

    rf_data = yf.download("^TNX", dt.datetime(2024, 8, 18), end)
    rf_data.to_csv('rf.csv')
    rf_file = open('rf.csv', 'r')
    lines = rf_file.readlines()
    del lines[0]
    rate_vector = []
    i = 0

    while i + 1 in range(len(lines)):
        rate = float(list(lines[i].strip('\n').split(COL_SEP))[COL_ADJ_CLOSE])
        rate_vector.append(rate * 0.1)
        i += 1
    summation = sum(rate_vector)
    return summation / (len(rate_vector)) * 0.01

# next thing to implement - beta estimate


def consecutive_days(date1: str, date2: str) -> bool:
    """Returns True if the dates given are in fact consecutive days. Otherwise, will return False.
    Helper function for the "one-step" logarithmic return calculator in order to preserve our one-step, as such
    we can assume the given dates will be in the same calendar year or the immediate following calendar year.

    Precondition: The dates are given in chronological order

    >>> consecutive_days('2023-09-27','2023-09-28')
    True
    >>> consecutive_days('2023-09-29','2023-10-02')
    False
    >>> consecutive_days('2023-10-31','2023-11-01')
    True
    """
    reformatted1 = dt.datetime.fromisoformat(date1)
    reformatted2 = dt.datetime.fromisoformat(date2)
    return dt.timedelta(days=1) == reformatted2-reformatted1


def average_return(ticker: str, start_date: str, method: str) -> tuple:
    """Returns the average return of a given stock from the selected start date to today as a float,
    as well as the return of each day from the start date to today. Method used to determine your average return
    is a required input.
    """

    day1 = dt.datetime.fromisoformat(start_date)
    stock_data = yf.download(ticker, day1, end)
    stock_data.to_csv(ticker + '.csv')
    stock_info = open(ticker + '.csv', 'r')

    lines = stock_info.readlines()
    del lines[0]
    sol_vector = []
    i = 0

    while i + 1 in range(len(lines)):
        line1 = list(lines[i].strip('\n').split(COL_SEP))
        line2 = list(lines[i + 1].strip('\n').split(COL_SEP))
        if consecutive_days(line1[0], line2[0]):
            if method == 'log':
                log_return = math.log(float(line2[COL_ADJ_CLOSE]) / float(line1[COL_ADJ_CLOSE]))
                sol_vector.append(log_return)
            if method == 'arithmetic':
                arithmetic_return = (float(line2[COL_ADJ_CLOSE])-float(line1[COL_ADJ_CLOSE]))/\
                                    float(line1[COL_ADJ_CLOSE])
                sol_vector.append(arithmetic_return)
        i += 1
    return sum(sol_vector)/len(sol_vector), sol_vector


def cost_of_carry_model(Futures: list, r: float, T: float) -> list:
    """Return an estimate of the spot price and residual c based on given futures contracts"""


def discount_curve()-> None:
    """"""

