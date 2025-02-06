

#!####################################################
#! Iyzico Transaction Volume Forecasting
#!####################################################

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')


#! Reading dataset and parsing according to transaction_date

df = pd.read_csv("D:/Çalışmalar/Miuul/Time Series/learning-space/iyzico_data.csv", parse_dates =['transaction_date'],index_col=0)
#! Checkout start and end date of the dataset

print(df['transaction_date'].max(), df['transaction_date'].min())

#! Checkout market place numbers in each category

print(df['merchant_id'].nunique())

#! Transaction amount of each unique merchants

print(df.groupby(['merchant_id']).agg({'Total_Transaction':'sum'}))


#! Observe merchants transaction count graphs 

df["transaction_date"] = pd.to_datetime(df["transaction_date"])

for id in df.merchant_id.unique():
    plt.figure(figsize=(15, 10))

    # 1. Grafik: 2018-2019 Transaction Count
    ax1 = plt.subplot(2, 1, 1)  # 2 satır, 1 sütun, 1. grafik
    df[(df['merchant_id'] == id) & 
       (df['transaction_date'] >= "2018-01-01") & 
       (df['transaction_date'] <= "2019-01-01")]['Total_Transaction'].plot(ax=ax1)
    ax1.set_title(f"{id} 2018-2019 Transaction Count")
    ax1.set_xlabel("")

    # 2. Grafik: 2019-2020 Transaction Count
    ax2 = plt.subplot(2, 1, 2)  # 2 satır, 1 sütun, 2. grafik
    df[(df['merchant_id'] == id) & 
       (df['transaction_date'] >= "2019-01-01") & 
       (df['transaction_date'] <= "2020-01-01")]['Total_Transaction'].plot(ax=ax2)
    ax2.set_title(f"{id} 2019-2020 Transaction Count")
    ax2.set_xlabel("")

    plt.tight_layout()  # Grafiklerin üst üste binmesini önler
    plt.show()



#! Creating Date Features

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df
