
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing





#! Durağanlık testi (Dickey-Fuller testi)
def is_stationary(y):

    # H0: Zaman serisi durağan değildir.
    # H1: Zaman serisi durağandır.

    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(f"Result: Stationary (H0: non stationary, p value: {round:p_value,3})")
    else:
        print(f"Result: Non Stationary (H0: stationary, p value: {round:p_value,3})")


#! Zaman serisi bileşenleri ve durağanlık testi
def ts_decompose(y, model='additive',stationary=False):
    result = seasonal_decompose(y, model=model)

    fig,axes = plt.subplots(4,1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title('Decomposition for ' + model + " model")
    axes[0].plot(y, 'k', label='Original' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal,'g', label='Seasonality & Mean : ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean : ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')

    plt.tight_layout()
    plt.show(block=True)

    if stationary:
        is_stationary(y)


def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title} - MAE: {mae:.2f}")
    test.plot(legend=True, label="TEST", figsize=(6,4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()


#! Single Exponential Smoothing Optimizer Function
def ses_optimizer(train, alphas, step=48):
    best_alpha, best_mae = None, float("inf")
    for alpha in alphas:
        model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae
        print("alpha:", round(alpha,2), "mae:", round(mae,2))
    print("Best alpha:", round(best_alpha,2), "Best MAE:", round(best_mae,2))
    return best_alpha, best_mae


#! Double Exponential Smoothing Optimizer Function
def des_optimizer(train, alphas, betas, step=48):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            model = ExponentialSmoothing(train, trend='add').fit(smoothing_level=alpha, smoothing_trend=beta)
            y_pred = model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha,2), "beta:", round(beta,2), "mae:", round(mae,4))
    print("Best alpha:", round(best_alpha,2), "Best beta:", round(best_beta,2), "Best MAE:", round(best_mae,4))
    return best_alpha, best_beta, best_mae




#! Triple Exponential Smoothing Optimizer Function
def tes_optimizer(train, alphas, betas, gammas, step=48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
                y_pred = model.forecast(step)
                mae = mean_absolute_error(test, y_pred)
                if mae < best_mae:
                    best_alpha, best_beta, best_gamma, best_mae = alpha, beta, gamma, mae
                print("alpha:", round(alpha,2), "beta:", round(beta,2), "gamma:", round(gamma,2), "mae:", round(mae,4))
    print("Best alpha:", round(best_alpha,2), "Best beta:", round(best_beta,2), "Best gamma:", round(best_gamma,2), "Best MAE:", round(best_mae,4))
    return best_alpha, best_beta, best_gamma, best_mae


def check_df(dataframe, head=5, tail=5, quan=False):
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



#! Creating time features

def create_date_features(dataframe):
    dataframe['MONTH'] = dataframe.date.dt.MONTH
    dataframe['DAY_OF_MONTH'] = dataframe.date.dt.DAY_OF_MONTH
    dataframe['DAY_OF_YEAR'] = dataframe.date.dt.DAY_OF_YEAR
    dataframe['WEEK_OF_YEAR'] = dataframe.date.dt.WEEK_OF_YEAR
    dataframe['DAY_OF_WEEK'] = dataframe.date.dt.DAY_OF_WEEK
    dataframe['YEAR'] = dataframe.date.dt.YEAR
    dataframe['IS_WEEKEND'] = dataframe.date.dt.weekday // 4
    dataframe['IS_MONTH_START'] = dataframe.date.dt.IS_MONTH_START.astype(int)
    dataframe['IS_MONTH_END'] = dataframe.date.dt.IS_MONTH_END.astype(int)

    return dataframe


#! lag_features
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' +str(lag)] = dataframe.groupby(['store','item'])['sales'].transform(
            lambda x:x.shift(lag)) + random_noise(dataframe)
    return dataframe


#! rolling mean windows
def roll_mean_feature(dataframe,windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(['store','item'])['sales'].\
            transform(lambda x:x.shift(1).rolling(window=window, min_periods=10, win_type="triang")).mean().\
                + random_noise(dataframe)
    return dataframe

 
#! Exponentially Weighted Features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".","")+"_lag_" + str(lag)] = \
                dataframe.groupby(["store",'item'])['sales'].transform(lambda x:x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

#! SMAPE calculation (Custom Cost Function)
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds==0) & (target==0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200*np.sum(num/denom)) / n
    return smape_val


#! SMAPE ile model değerlendirme
def lgbm_smape(preds,train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


#! feature importances
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


#! ARIMA Optimizer Function
def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arima_model_result = ARIMA(train, order = order).fit()
            aic = arima_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print("ARIMA%s AIC=%.2f" % (order, aic))
        except:
            continue
    print("Best ARIMA%s AIC=%.2f" % (best_params, best_aic))
    return best_params


#! SARIMA Optimizer Function
def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order= param, seasonal_order=param_seasonal)
                results = sarimax_model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print("SARIMA{}x{}12 - AIC:{}".format(param, param_seasonal, aic))
            except:
                continue
    print("SARIMA {}x{}12 - AIC:{}".format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order


#! Sarima Optimizer Function with MAE
def sarima_optimizer_mae(train, pdq, seasonal_pdq):
best_mae, best_order, best_seasonal_order = float("inf"), float("inf"), None
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
            sarima_model = model.fit()
            y_pred_test = sarima_model.get_forecast(steps=48)
            y_pred = y_pred_test.predicted_mean
            mae = mean_absolute_error(test, y_pred)

            #mae = fit_model_sarima(train, val, param, param_seasonal)

            if mae < best_mae:
                best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
            print("SARIMA{}x{}12 - MAE:{}".format(param, param_seasonal, mae))
        except:
            continue
print("SARIMA{}x{}12 - MAE:{}".format(best_order, best_seasonal_order, best_mae))
return best_order, best_seasonal_order

#! Plotting Prediction
def plot_prediction(y_pred, label):
    train['total_passengers'].plot(legend=True, label='TRAIN')
    test['total_passengers'].plot(legend=True, label='TEST')
    y_pred.plot(legend=True, label='PREDICTION')
    plt.title('Train, Test and Predicted Test using {}'.format(label))


#! Creating date features
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

#! Creating lag/shifted features
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe



#! Creating Exponentially Weighted Mean Features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

#! Smape
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