# I am now starting to realize that oop would have been a better approach to this
# but hey, functional programming is always a fun time

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import create_engine
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# We will use volvisualizer for nice visualizations

# Intiializing access to the options database
user = "root"            # default user
password = ""            # blank if no password
host = "localhost"
port = 3306
db_name = "options"

# Create SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}")

def get_earnings_dates():
    df = pd.read_csv("/Users/lunesm/School Stuff/Projects/EPS Option Reaction/earnings_dates.csv")
    earnings_dict = {}
    for ticker in df.columns:
        dates = df[ticker].dropna().astype(str).tolist()
        earnings_dict[ticker] = dates
    return earnings_dict

earnings_dates = get_earnings_dates()

def fetch_atm_option_data(ticker, engine):
    results = []
    for date in earnings_dates[ticker]:
        try:
            date_dt = datetime.strptime(date, "%Y-%m-%d")
            buy_date = (date_dt - 2 * BUSINESS_DAY).strftime("%Y-%m-%d")
            sell_date = (date_dt + BUSINESS_DAY).strftime("%Y-%m-%d")
            spot_price = yf.download(ticker, start= buy_date,
                                     end=sell_date, interval="1d")
        
            spot_price = spot_price['Close'].values[0][0]                    

            query_buy = f"""
                SELECT *
                FROM `option_chain`
                WHERE
                    `act_symbol` = '{ticker}'
                    AND `call_put` = 'Call'
                    AND `date` = '{buy_date}'
                    AND `expiration` BETWEEN DATE_ADD('{buy_date}', INTERVAL 27 DAY)
                                 AND DATE_ADD('{buy_date}', INTERVAL 33 DAY)
                ORDER BY ABS(`strike` - {spot_price}) ASC, `expiration` ASC
                LIMIT 1;
                """
        
            df_buy = pd.read_sql(query_buy, engine)
        
            if df_buy.empty:
                raise ValueError("No buy data")
        
            contract = df_buy.iloc[0]
            strike = contract['strike']
            expiration = contract['expiration']
            buy_ask = contract['ask']
            vol = contract['vol']
            vega = contract['vega']


            query_sell = f"""
                SELECT *
                FROM `option_chain`
                WHERE
                    `act_symbol` = '{ticker}'
                    AND `call_put` = 'Call'
                    AND `date` = '{sell_date}'
                    AND `expiration` = '{expiration}'
                AND strike = {strike}
                LIMIT 1;
                """
        
            df_sell = pd.read_sql(query_sell, engine)

            sell_bid = df_sell.iloc[0]['bid']
            option_return = (sell_bid - buy_ask) / buy_ask

            results.append({
                "earnings_date": date,
                "volatility" : vol,
                "vega" : vega,
                "buy_date": buy_date,
                "sell_date": sell_date,
                "strike": strike,
                "expiration": expiration,
                "spot_price": spot_price,
                "buy_ask": buy_ask,
                "option_return": option_return
            })

        except Exception as e:
            results.append({
                "earnings_date": date,
                "volatility": np.nan,
                "vega": np.nan,
                "buy_date": buy_date if 'buy_date' in locals() else np.nan,
                "sell_date": sell_date if 'sell_date' in locals() else np.nan,
                "strike": np.nan,
                "expiration": np.nan,
                "spot_price": np.nan,
                "buy_ask": np.nan,
                "option_return": np.nan
                })

    return pd.DataFrame(results)
    
    

universe = ['AAPL','MSFT','GOOGL','AMZN','AVGO','IBM','ORCL','ADBE', "CRM", "NVDA"]

# Extract step


def get_y(ticker):
    # Okay
    # This would typically involve using the actual stock price movement after earnings
    return 0.0

def get_momentum_vol(ticker, window_momentum=5, window_vol=21):
    df = []
    dates = get_earnings_dates()[ticker]

    for earnings_date in dates:
        earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
        # Shift to 2 trading days before earnings
        lookback_end = earnings_dt - (2 * BUSINESS_DAY)
        # Pull 60 calendar days of data before that
        lookback_start = lookback_end - (60 * BUSINESS_DAY)
        data = yf.download(ticker, start=lookback_start.strftime("%Y-%m-%d"), end=lookback_end.strftime("%Y-%m-%d"), interval="1d")
        data = data.dropna()
        if len(data) < max(window_momentum, window_vol):
            continue  # skip if not enough data
        # Calculate features using safe slice
        momentum = data['Close'].pct_change(window_momentum).iloc[-1].item()
        returns = data['Close'].pct_change().dropna()
        realized_vol = (returns[-window_vol:].std() * (252**0.5)).item()
        avg_volume = data['Volume'][-window_vol:].mean().item()
        df.append({
            "earnings_date": earnings_date,
            "momentum": momentum,
            "realized_vol": realized_vol,
            "avg_volume": avg_volume
        })
    return pd.DataFrame(df)

def extract_data():
    data = {}
    for ticker in universe:
        data[ticker] = {}
        data[ticker]['momentum_vol'] = get_momentum_vol(ticker)
        data[ticker]['implied_vol'] = ...
        data[ticker]['probability_surprise'] = ...
        data[ticker]['vega'] =...
    return data

# Load

def get_Y():
    ...

def get_X():
    ...

def clean_Y():
    ...

def clean_X():
    
# We will need to generate an X and y dataset for the model




