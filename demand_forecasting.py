
#! ####################################3
#! Demand Forecasting
#! ####################################3

#! Store Item Demand Forecasting Challenge

#! 3 aylık ürün tahmin talep

import time
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns
import lightgbm as lgb 

pd.set_option('display.max_columns',None)




def check_df(dataframe, head=5, tail=5, quan=True):
    print('########## Shape ##########')
    print(dataframe.shape)
    print('########## Types ##########')
    print(dataframe.dtypes)
    print("##################### Duplicated Values #####################")
    print(dataframe.duplicated().sum())
    print('########## Head ##########')
    print(dataframe.head(head))
    print('########## Tail ##########')
    print(dataframe.tail(tail))
    print('########## NA ##########')
    print(dataframe.isnull().sum())
    
    if quan:
        print('########## Quantiles ##########')
        print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


#!######################################
#! Loading the data
#!######################################

train = pd.read_csv("D:/Çalışmalar/Miuul/Time Series/time_series/datasets/demand_forecasting/train.csv", parse_dates=['date'])
test = pd.read_csv("D:/Çalışmalar/Miuul/Time Series/time_series/datasets/demand_forecasting/test.csv",parse_dates=['date'])

sample_sub = pd.read_csv("D:/Çalışmalar/Miuul/Time Series/time_series/datasets/demand_forecasting/sample_submission.csv")

df = pd.concat([train,test], sort=False)


#!#############################3
#! EDA
#!#############################3

df['date'].min(), df['date'].max()

check_df(df)

df[['store']].nunique()

df[['item']].nunique()

df.groupby(['store'])['item'].nunique()

df.groupby(['store','item']).agg({'sales':sum}

#! Mağaza ürün kırılımında satış istatistikleri

df.groupby(['store','item']).agg({'sales': ['sum','mean','median','std']})

df.head()


#!###########################################
#! FEATURE ENGINEERING (EN ONEMLI NOKTA)
#!###########################################

def create_date_features(dataframe):
    dataframe['MONTH'] = dataframe.date.dt.month
    dataframe['DAY_OF_MONTH'] = dataframe.date.dt.day
    dataframe['DAY_OF_YEAR'] = dataframe.date.dt.dayofyear
    dataframe['WEEK_OF_YEAR'] = dataframe.date.dt.isocalendar().week  # Güncel kullanım
    dataframe['DAY_OF_WEEK'] = dataframe.date.dt.dayofweek
    dataframe['YEAR'] = dataframe.date.dt.year
    dataframe['IS_WEEKEND'] = (dataframe.date.dt.weekday >= 5).astype(int)  # Daha açıklayıcı
    dataframe['IS_MONTH_START'] = dataframe.date.dt.is_month_start.astype(int)
    dataframe['IS_MONTH_END'] = dataframe.date.dt.is_month_end.astype(int)

    return dataframe


df = create_date_features(df)

df.head()


#!###########################################
#! Random Noise
#!###########################################  

# Aşırı öğrenmenin önüne geçmek için rastgele gürültü eklemeliyim.
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


#!###########################################
#! Lag / Shifted Features
#!###########################################  

#!Geçmiş dönem satış sayılarına ilişkin featurelar türeteceğiz.

df.sort_values(by=['store','item','date'],axis=0,inplace=True)

pd.DataFrame({"sales":df["sales"].values[0:10],
            "lag1": df["sales"].shift(1).values[0:10],
            "lag2": df["sales"].shift(2).values[0:10],
            "lag3": df["sales"].shift(3).values[0:10],
            "lag4": df["sales"].shift(3).values[0:10]})

df.groupby(['store','item'])['sales'].head()

df.groupby(['store','item'])['sales'].transform(lambda x:x.shift(1))


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' +str(lag)] = dataframe.groupby(['store','item'])['sales'].transform(
            lambda x:x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df,[91,98,105,112,119,126,182,364,546,728])

check_df(df)

df.tail()




#!###############################################
#! Rolling Mean Features
#!###############################################

pd.DataFrame({"sales":df["sales"].values[0:10],
            "roll2":df["sales"].rolling(window=2).mean().values[0:10],
            "roll3":df["sales"].rolling(window=3).mean().values[0:10],
            "roll4":df["sales"].rolling(window=4).mean().values[0:10],
            "roll5":df["sales"].rolling(window=5).mean().values[0:10]})


pd.DataFrame({"sales":df["sales"].values[0:10],
            "roll2":df["sales"].shift(1).rolling(window=2).mean().values[0:10],
            "roll3":df["sales"].shift(1).rolling(window=3).mean().values[0:10],
            "roll4":df["sales"].shift(1).rolling(window=4).mean().values[0:10],
            "roll5":df["sales"].shift(1).rolling(window=5).mean().values[0:10]})


def roll_mean_feature(dataframe,windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(['store','item'])['sales'].\
            transform(
            lambda x:x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(dataframe)
    return dataframe

df = roll_mean_feature(df, [365, 546])

df.head()


#!#######################################
#! Exponentially Weighted Mean Features
#!#######################################

pd.DataFrame({"sales":df['sales'].values[0:10],
            "roll2":df["sales"].shift(1).rolling(window=2).mean().values[0:10],
            "ewm099":df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
            "ewm095":df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
            "ewm07":df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
            "ewm02":df["sales"].shift(1).ewm(alpha=0.2).mean().values[0:10]})


def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".","")+"_lag_" + str(lag)] = \
                dataframe.groupby(["store",'item'])['sales'].transform(lambda x:x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.7, 0.5] 
lags = [91,98, 105,112,180,270,365,546,728]

df = ewm_features(df,alphas,lags)

check_df(df)

#!#######################################
#! One-Hot Encoding
#!#######################################


df = pd.get_dummies(df, columns=['store','item','DAY_OF_WEEK','MONTH'])

#!##############################################
#! Converting Sales to log(1+sales)
#!##############################################

#? Train süresinin daha kısa süre alması için

df['sales'] = np.log1p(df['sales'].values)

check_df(df)



#!##############################################
#! Custom Cost Function
#!##############################################

#? MAE, MSE, RMSE, SSE
#? MAPE: mean absolute percentage error
#? SMAPE: symmetric mean absolute percentage error (adjusted MAPE)
#? SMAPE ne kadar düşük o kadar iyi

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds==0) & (target==0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200*np.sum(num/denom)) / n
    return smape_val


def lgbm_smape(preds,train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False



#!##############################################
#! Time-Based Validation Sets
#!##############################################

train
test
#2017'nin başın (2016 sonuna kadar) train seti
train = df.loc[(df['date']< "2017-01-01"),:]


#2017'nin ilk 3 ayı validasyon seti
val = df.loc[(df['date']>="2017-01-01") & (df["date"] < "2017-04-01"),:]

cols = [col for col in train.columns if col not in ['date','id','sales','year']]

y_train = train['sales']
X_train = train[cols]

y_val = val['sales']
X_val = val[cols]


y_train.shape,X_train.shape, y_val.shape, X_val.shape


#!##############################################
#! LightGBM ile Zaman Serisi Modeli
#!##############################################

#! LightGBM parameters

lgb_params = {'num_leaves':10,
            'learning_rate':0.02,
            'feature_fraction':0.8,
            'max_depth':5,
            'verbose':0,
            'num_boost_round':10000,  #iterasyon sayısı, optimizasyon sayısı,10000 olması iyidir.
            'early_stopping_rounds':500,
            'nthread':-1}

#metric mae: l1, absoulte loss, mean_absolute error, regression_l1,
# mse, l2, square_loss, mean_squared_error, mse, regression_l1, regression
# rmse, root square loss, root_mean_squared_error, l2_root
#mape, MAPE loss, mean_absolute_percentage_error

#? num_leaves: bir ağaçtaki maksimum yaprak sayısı
#? learning_rate: shrinkage_rate, eta
#? feature_fraction: rf'nin random subspace özelliği. Her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı
#? max_depth: maksimum derinlik
#? num_boost_round: n_estimators: number of boosting iterations. En az 10000-150000 civarı yapmak lazım.abs

#? early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_Rounds'da ilerlemiyorsa yan
#? hata düşmüyorsa modellemyi durdur.
#? hem train süresini kısaltır hem de overfit'e engel olur.
#? nthread: num_thread, nthread, n_jobs: İşlemcilerin tamamını full performans kullanmak için parametre.


#! Lightgbm dataset formatında daha hızlı train etme imkanımız var.
lgbtrain = lgb.Dataset(data=X_train,label=y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=y_val, reference=lgbtrain, feature_name=cols)


callbacks = [
    lgb.early_stopping(stopping_rounds=lgb_params['early_stopping_rounds']),
    lgb.log_evaluation(period=100)
]

model = lgb.train(
    lgb_params,
    lgbtrain,
    valid_sets=[lgbtrain, lgbval],
    num_boost_round=lgb_params['num_boost_round'],
    feval=lgbm_smape,
    callbacks=callbacks)


y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val),np.expm1(y_val))


#!###################################
#! Değişken önem düzeyleri
#!###################################

def plot_lgb_importances(model, plot=False,num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature':model.feature_name(),
                            'split':model.feature_importance('split'),
                            'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    
    if plot:
        plt.figure(figsize=(10,10))
        sns.set(font_scale = 1)
        sns.barplot(x="gain", y ="feature",data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    
    return feat_imp 

plot_lgb_importances(model, num=30)

feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp['gain'] == 0]['feature'].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)

#!###################################
#! Final Model
#!###################################

train = df.loc[~df.sales.isna()]

y_train = train['sales']
X_train = train[imp_feats]


test = df.loc[df.sales.isna()]
X_test = test[imp_feats]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction':0.8,
              'max_depth':5,
              'verbose':0,
              'nthread':-1,
              'num_boost_round':model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_test, num_iteration = model.best_iteration)

#!###################################
#! Submission File
#!###################################

test.head()

submission_df = test.loc[:,['id','sales']]

submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] =submission_df['id'].astype(int)

submission_df.to_csv("submission_demand.csv",index=False)