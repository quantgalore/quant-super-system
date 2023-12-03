# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 01:34:34 2023

@author: Locale
"""

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from sklearn.ensemble import RandomForestClassifier
from feature_functions import Binarizer, return_proba

import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import sqlalchemy
import mysql.connector
import sys


polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
sector_dataset = pd.read_sql(sql = "SELECT * FROM sp500_sector_dataset", con = engine).set_index("t")


exchange = 'NYSE'
calendar = get_calendar(exchange)
date = datetime.today().strftime("%Y-%m-%d")

valid_dates = calendar.schedule(start_date = (datetime.today() - timedelta(days = 5)).strftime("%Y-%m-%d"), end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")).index.strftime("%Y-%m-%d").values

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
                    xlu.add_prefix("Utilities_")], axis = 1).dropna()

Merged_Returns = Merged[["Communications_returns","ConsumerDiscretionary_returns", "ConsumerStaples_returns", "Energy_returns",
                         "Financials_returns", "Healthcare_returns", "Industrials_returns",
                         "Materials_returns","RealEstate_returns", "Technology_returns",
                         "Utilities_returns"]].copy()


Training_Dataset = sector_dataset[sector_dataset.index < pd.to_datetime(date)].copy().tail(100)

# train the model

X_Classification = Training_Dataset.drop("SP500_returns", axis = 1).values
Y_Classification = Training_Dataset["SP500_returns"].apply(Binarizer).values.ravel()

RandomForest_Classification_Model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None).fit(X_Classification, Y_Classification)

SP500_Production_Dataset = Merged_Returns.tail(1).values

Classification_Prediction = RandomForest_Classification_Model.predict(SP500_Production_Dataset)

Classification_Prediction_Probability = RandomForest_Classification_Model.predict_proba(SP500_Production_Dataset)

random_forest_prediction_dataframe = pd.DataFrame({"prediction": Classification_Prediction})
random_forest_prediction_dataframe["probability_0"] = Classification_Prediction_Probability[:,0]
random_forest_prediction_dataframe["probability_1"] = Classification_Prediction_Probability[:,1]
random_forest_prediction_dataframe["probability"] = return_proba(random_forest_prediction_dataframe)

probability = random_forest_prediction_dataframe["probability"].iloc[0]

Prediction = Classification_Prediction[0]

if probability <= .65:
    print("Probability too low.")
    sys.exit()

base_url = 'https://api.tastyworks.com'

# authenticate session

auth_url = 'https://api.tastyworks.com/sessions'
headers = {'Content-Type': 'application/json'}

session_data = {
    "login": "tastytrade@login.com",
    "password": "password",
    "remember-me": True
}

authentication_response = requests.post(auth_url, headers=headers, json=session_data)
authentication_json = authentication_response.json()

session_token = authentication_json["data"]["session-token"]
authorized_header = {'Authorization': session_token}

# DO NOT SPAM ^^
accounts = requests.get(f"{base_url}/customers/me/accounts", headers = {'Authorization': session_token}).json()
account_number = accounts["data"]["items"][0]["account"]["account-number"]

balances = requests.get(f"{base_url}/accounts/{account_number}/balances", headers = {'Authorization': session_token}).json()["data"]

option_url = f"https://api.tastyworks.com/option-chains/SPY/nested"

eod_price = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"])["c"].iloc[0]

option_chain = pd.json_normalize(requests.get(option_url,  headers = {'Authorization': session_token}).json()["data"]["items"][0]["expirations"][1]["strikes"])
option_chain["strike-price"] = option_chain["strike-price"].astype(float)
option_chain["distance_from_price"] = abs(option_chain["strike-price"] - eod_price)

atm_strike = option_chain[option_chain["distance_from_price"] == option_chain["distance_from_price"].min()]

if Prediction == 1:
    option_ticker = atm_strike["call"].iloc[0]
elif Prediction == 0:
    option_ticker = atm_strike["put"].iloc[0]

order_details = {
    "time-in-force": "Day",
    "order-type": "Market",
    "price-effect": "Debit",
    "legs": [{
          "instrument-type": "Equity Option",
          "symbol": f"{option_ticker}",
          "quantity": 1,
          "action": "Buy to Open"
              }]
                }

validate_order = requests.post(f"https://api.tastyworks.com/accounts/{account_number}/orders/dry-run", json = order_details, headers = {'Authorization': session_token})
validation_text = validate_order.json()

submit_order = requests.post(f"{base_url}/accounts/{account_number}/orders", json = order_details, headers = {'Authorization': session_token})
order_submission_text = submit_order.json()

# Demo Sandbox

demo_base_url =f"https://api.cert.tastyworks.com"
demo_auth_url = 'https://api.cert.tastyworks.com/sessions'

demo_credentials = {"login": "tastytrade-sandbox-name","password": "sandbox-password","remember-me": True}
demo_authentication_response = requests.post(demo_auth_url, json=demo_credentials)
demo_authentication_json = demo_authentication_response.json()

demo_session_token = demo_authentication_json["data"]["session-token"]
demo_authorized_header = {'Authorization': demo_session_token}

demo_accounts = requests.get(f"{demo_base_url}/customers/me/accounts", headers = {'Authorization': demo_session_token}).json()
demo_account_number = demo_accounts["data"]["items"][0]["account"]["account-number"]

demo_balances = requests.get(f"{demo_base_url}/accounts/{demo_account_number}/balances", headers = {'Authorization': demo_session_token}).json()["data"]

order_details = {
    "time-in-force": "Day",
    "order-type": "Market",
    "price-effect": "Debit",
    "legs": [{
          "instrument-type": "Equity",
          "symbol": f"SPY",
          "quantity": 1,
          
          "action": "Buy to Open"
              }]
                }

validate_demo_order = requests.post(f"{demo_base_url}/accounts/{demo_account_number}/orders/dry-run", json = order_details, headers = {'Authorization': demo_session_token})
demo_validation_text = validate_demo_order.json()

validation_status = validate_demo_order.reason

if validation_status == "Created":
    # order passed pre-checks (e.g., enough capital)
    demo_submit_order = requests.post(f"{demo_base_url}/accounts/{demo_account_number}/orders", json = order_details, headers = {'Authorization': demo_session_token})
    demo_order_status = demo_submit_order.reason
    if demo_order_status == "Created":
        print(f"Order Succesfully Submitted.")
        