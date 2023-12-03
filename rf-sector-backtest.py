# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:53:26 2023

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

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
sector_dataset = pd.read_sql(sql = "SELECT * FROM sp500_sector_dataset", con = engine).set_index("t")

# we get the valid dates where the market was open

exchange = 'NYSE'
calendar = get_calendar(exchange)

valid_dates = calendar.schedule(start_date = "2023-01-01", end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")).index.strftime("%Y-%m-%d").values

prediction_actual_list = []

for trade_date in valid_dates:
    
    try:
        
        start_time = datetime.now()
        
        if trade_date == valid_dates[-1]:
            continue
        
        next_day = valid_dates[(np.where(valid_dates == trade_date)[0][0])+1]
        
        # get the actual overnight return to later compare
        
        actual_return = sector_dataset[sector_dataset.index.strftime("%Y-%m-%d") == trade_date]
    
        if len(actual_return) < 1:
            continue
    
        Training_Dataset = sector_dataset[sector_dataset.index < pd.to_datetime(trade_date)].copy().tail(100)
        
        # train the model
        
        X_Classification = Training_Dataset.drop("SP500_returns", axis = 1).values
        Y_Classification = Training_Dataset["SP500_returns"].apply(Binarizer).values.ravel()

        RandomForest_Classification_Model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None).fit(X_Classification, Y_Classification)
        
        #### Production 
        
        SP500_Production_Dataset = actual_return.drop("SP500_returns", axis = 1).values
        
        Classification_Prediction = RandomForest_Classification_Model.predict(SP500_Production_Dataset)
        
        Classification_Prediction_Probability = RandomForest_Classification_Model.predict_proba(SP500_Production_Dataset)

        random_forest_prediction_dataframe = pd.DataFrame({"prediction": Classification_Prediction})
        random_forest_prediction_dataframe["probability_0"] = Classification_Prediction_Probability[:,0]
        random_forest_prediction_dataframe["probability_1"] = Classification_Prediction_Probability[:,1]
        random_forest_prediction_dataframe["probability"] = return_proba(random_forest_prediction_dataframe)
        
        probability = random_forest_prediction_dataframe["probability"].iloc[0]
        
        Prediction = Classification_Prediction[0]
        
        # get trade price
        
        eod_price = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{trade_date}/{trade_date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"])["c"].iloc[0]
        
        if Prediction == 0:
            Put_Contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker=SPY&contract_type=put&expiration_date={next_day}&as_of={trade_date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
            Put_Contracts["distance_from_price"] = abs(Put_Contracts["strike_price"] - eod_price)
            
            ATM_Strike = Put_Contracts[Put_Contracts["distance_from_price"] == Put_Contracts["distance_from_price"].min()].copy()
        elif Prediction == 1:        
            Call_Contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker=SPY&contract_type=call&expiration_date={next_day}&as_of={trade_date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
            Call_Contracts["distance_from_price"] = abs(Call_Contracts["strike_price"] - eod_price)
            
            ATM_Strike = Call_Contracts[Call_Contracts["distance_from_price"] == Call_Contracts["distance_from_price"].min()].copy()
        
        open_price = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ATM_Strike['ticker'].iloc[0]}/range/1/day/{trade_date}/{trade_date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")["c"].iloc[0]
        
        if (probability <= .65) or (open_price > 1.5):
            continue
        
        closing_price = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ATM_Strike['ticker'].iloc[0]}/range/1/day/{next_day}/{next_day}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")["o"].iloc[0]
        
        gross_pnl = closing_price - open_price
        
        Pred_Actual = pd.DataFrame([{"prediction_date": pd.to_datetime(trade_date),
                                    "classification_prediction": Classification_Prediction[0],
                                    "probability": probability,
                                  "actual": actual_return["SP500_returns"].iloc[0],
                                  "actual_binary": Binarizer(actual_return["SP500_returns"].iloc[0]),
                                    "open_price": open_price, "closing_price": closing_price,
                                    "gross_pnl":gross_pnl}])
        
        prediction_actual_list.append(Pred_Actual)
        
        end_time = datetime.now()
        iteration = round((np.where(valid_dates==trade_date)[0][0]/len(valid_dates))*100,2)
        iterations_remaining = len(valid_dates) - np.where(valid_dates==trade_date)[0][0]
        average_time_to_complete = (end_time - start_time).total_seconds()
        estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
        time_remaining = estimated_completion_time - datetime.now()
        
        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")

    except Exception as error:
        print(error)
        continue
    
complete_trades_original = pd.concat(prediction_actual_list).set_index("prediction_date")

complete_trades = complete_trades_original.copy()
complete_trades["capital"] = 1000 + complete_trades["gross_pnl"].cumsum() * 100

plt.figure(dpi = 600)
plt.xticks(rotation = 45)
plt.suptitle(f"Strategy: 1x call/put when confidence > 65%")
plt.title("01/23 - 12/23")
plt.plot(complete_trades["capital"])
plt.show()
