# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

import pandas as pd
import numpy as np
import requests
import sqlalchemy
import mysql.connector
    
polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"

# we get the valid dates where the market was open

exchange = 'NYSE'
calendar = get_calendar(exchange)

valid_dates = calendar.schedule(start_date = "2018-04-26", end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")).index.strftime("%Y-%m-%d").values

spy = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
spy.index = pd.to_datetime(spy.index, unit = "ms", utc = True).tz_convert("America/New_York")

xlc = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/XLC/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
xlc.index = pd.to_datetime(xlc.index, unit = "ms", utc = True).tz_convert("America/New_York")

xly = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/XLY/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
xly.index = pd.to_datetime(xly.index, unit = "ms", utc = True).tz_convert("America/New_York")

xlp = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/XLP/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
xlp.index = pd.to_datetime(xlp.index, unit = "ms", utc = True).tz_convert("America/New_York")

xle = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/XLE/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
xle.index = pd.to_datetime(xle.index, unit = "ms", utc = True).tz_convert("America/New_York")

xlf = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/XLF/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
xlf.index = pd.to_datetime(xlf.index, unit = "ms", utc = True).tz_convert("America/New_York")

xlv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/XLV/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
xlv.index = pd.to_datetime(xlv.index, unit = "ms", utc = True).tz_convert("America/New_York")

xli = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/XLI/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
xli.index = pd.to_datetime(xli.index, unit = "ms", utc = True).tz_convert("America/New_York")

xlb = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/XLB/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
xlb.index = pd.to_datetime(xlb.index, unit = "ms", utc = True).tz_convert("America/New_York")

xlre = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/XLRE/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
xlre.index = pd.to_datetime(xlre.index, unit = "ms", utc = True).tz_convert("America/New_York")

xlk = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/XLK/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
xlk.index = pd.to_datetime(xlk.index, unit = "ms", utc = True).tz_convert("America/New_York")

xlu = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/XLU/range/1/day/{valid_dates[0]}/{valid_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
xlu.index = pd.to_datetime(xlu.index, unit = "ms", utc = True).tz_convert("America/New_York")

etf_list = [xlc, xly, xlp, xle, xlf, xlv, xli, xlb, xlre, xlk, xlu]

for etf in etf_list:
    etf["returns"] = etf["c"].pct_change().fillna(0) * 100

Merged = pd.concat([xlc.add_prefix("Communications_"),
                    xly.add_prefix("ConsumerDiscretionary_"),
                    xlp.add_prefix("ConsumerStaples_"), xle.add_prefix("Energy_"),
                    xlf.add_prefix("Financials_"), xlv.add_prefix("Healthcare_"),
                    xli.add_prefix("Industrials_"), xlb.add_prefix("Materials_"),
                    xlre.add_prefix("RealEstate_"),xlk.add_prefix("Technology_"),
                    xlu.add_prefix("Utilities_"), spy.add_prefix("SP500_")], axis = 1).dropna()

Merged["SP500_returns"] = ((Merged["SP500_o"] - Merged["SP500_c"].shift(1)) / Merged["SP500_c"].shift(1)).fillna(0) * 100

Merged_Returns = Merged[["Communications_returns","ConsumerDiscretionary_returns", "ConsumerStaples_returns", "Energy_returns",
                         "Financials_returns", "Healthcare_returns", "Industrials_returns",
                         "Materials_returns","RealEstate_returns", "Technology_returns",
                         "Utilities_returns", "SP500_returns"]].copy()

Shifted_Merged_Returns = Merged_Returns.copy()

# when sector x moved by x%, SPY opened up x% lower/higher the next day

Shifted_Merged_Returns["SP500_returns"] = Shifted_Merged_Returns["SP500_returns"].shift(-1).dropna()
Shifted_Merged_Returns = round(Shifted_Merged_Returns.dropna(), 2)

engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
Shifted_Merged_Returns.to_sql("sp500_sector_dataset", con = engine, if_exists = "replace")