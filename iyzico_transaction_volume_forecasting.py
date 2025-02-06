

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
import re

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

def create_date_features(df, date_column):
    df['month'] = df[date_column].dt.month
    df['day_of_month'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['year'] = df[date_column].dt.year
    df["is_wknd"] = df[date_column].dt.weekday // 4
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    return df

df = create_date_features(df, "transaction_date")



#! Creating lag/shifted features


def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df,[91, 182, 365, 541, 720])


#! Creating Rolling Mean Features

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["merchant_id"])['Total_Transaction']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df,[91, 182, 365, 541, 720])


#! Creating Exponentially Weighted Mean Features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 182, 365, 541, 720]

df = ewm_features(df, alphas, lags)


#! Special days

df['is_black_friday'] = 0

df.loc[df['transaction_date'].isin(['2018-11-22','2018-11-23','2019-11-29','2018-11-30']),'is_black_friday'] = 1

df['is_summer_solstice'] = 0

df.loc[df['transaction_date'].isin(['2018-06-19','2018-06-20','2018-06-21','2018-06-22',
                                    '2019-06-19','2019-06-19','2019-06-19','2019-06-19']),'is_summer_solstice'] = 1



#! One - hot encoding

df = pd.get_dummies(df, columns=['merchant_id','day_of_week', 'month'])
df['Total_Transaction'] = np.log1p(df['Total_Transaction'].values)


#! Defining custom cost functions

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


#! Creation of time based train and validation sets

df = df.rename(columns= lambda x:re.sub('[^A-za-z0-9_]+', '', x))

#! train till the end of tenth month of 2020

train = df.loc[(df['transaction_date']< '2020-10-01'),:]

#! val -> last three months of 2020

val = df.loc[(df['transaction_date'] >= '2020-10-01'),:]

cols = [col for col in train.columns if col not in ['transaction_date','id', 'Total_Transaction','Total_paid','year']]

Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]

#!#####################
#! Lightgbm model
#!#####################

lgb_params = {'metric': {'mae'},
              'num_leaves':10,
              'learning_rate':0.02,
              'feature_fraction':0.8,
              'max_depth':5,
              'verbose':0,
              'num_boost_round':1000,
              'nthread':-1}

lgbtrain = lgb.Dataset(data=X_train,label=Y_train,feature_name=cols)
lgbval = lgb.Dataset(data=X_val,label=Y_val,feature_name=cols)


model = lgb.train(lgb_params,lgbtrain,
                valid_sets=[lgbtrain, lgbval],
                num_boost_round=lgb_params['num_boost_round'],
                feval=lgbm_smape)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))


#! Feature Importances

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, plot=True, num = 20)