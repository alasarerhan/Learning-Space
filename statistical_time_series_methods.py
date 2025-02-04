

#AR(p) Autoregression
#! A time series is predicted with previous lags.

#! It is estimated with a linear combination of observations from previous time steps.
#! Suitable for univariate time series without trend and seasonality.
#! p: is the number of time steps. p = 1 If p = 1 is 1, it means that the model is established with the previous time step.

#! p = 1 -> yt = a1*yt-1 + et
#! p = 2 -> yt = a1*yt-1 + a2*yt-2 + et


#! MA(q) Moving Average
#yt = m1*et-1 + et
#! yt = m1*et-1 + m2*et-2 + .... + mq*et-q + et
#! Forecast with a linear combination of errors from previous time steps.
#! Suitable for univariate time series without trend and seasonality.
#! q: It is the number of time lags. 

#! ARMA(p,q) = AR(p) + MA(q)
#! yt = a1*yt-1 + m1*et-1 + et
#! ARMA is similar to Simple Exponential Smoothing. But ARMA is in the form of a linear regression.
#! While in Holt Winters methods the terms are shaped according to a parameter, in AR, MA and  ARMA models the terms have their own coefficients.
#! We are interested in finding these coefficients. 
#! Aurotregressive Moving Average. Combines AR and MA methods.
#! Forecasting is done with a linear combination of past values and past errors.
#! Suitable for univariate time series without trends and seasonality.
#! p and q are the number of time lags. p is the number of lags required for the AR model and q is the number of lags required for the MA model.



#! ARIMA(p,d,q) Autoregressive Integrated Moving Average
"""
By focusing on past actual values and past errors (residuals), the modeling process takes place in a linear regression form that will reveal the coefficients of the AR and MA terms in a linear form.

In ARIMA, this is done after a differencing process.
ARIMA can model trend and seasonality. 

Forecasting is done with a linear combination of differenced observations and errors from previous time steps.

When we stationarize a series, we assume that we can then forecast more successfully, so the ARIMA method automatically performs differencing to stationarize the series.

ARIMA is suitable for data with univariate trend but no seasonality.
p: number of real value lags (autoregressive degree) If p = 2, yt-1 and yt-2 are in the model.
d: number of difference operations (difference degree, I)
q: number of error lags (moving average degree)
"""

#?################################
#? ARIMA MODEL
#?################################
import matplotlib.pyplot as plt
import itertools
import warnings
import numpy as np
import pandas as pd 
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.holtwinters import ExponentialSmoothing
#from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt 
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


#############################
# DATASET
#############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas()
y = data.data 
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())
train = y[:'1997-12-01']
test = y['1998-01-01':]

#################################################################################
# ARIMA(p,d,q): (Autoregressive Integrated Moving Average)
#################################################################################

#!For AR, MA and ARMA models, the data set should be stationary.
#! Not suitable for data with seasonality and trends
#! The ARIMA model can also be used in trended series by taking the difference.
arima_model = ARIMA(train, order = (1,1,1)).fit()


#? Statistical Output of the model
print(arima_model.summary())

#? Predictions
y_pred = arima_model.forecast(48)[0]
print(y_pred)

#? Visualization

def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title} - MAE: {mae:.2f}")
    test.plot(legend=True, label="TEST", figsize=(6,4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()


y_pred = pd.Series(y_pred, index = test.index)

plot_co2(train, test, y_pred, "ARIMA")

#! The ARIMA model is a sibling of the DES (Double Exponential Smoothing) model.
#! DES can model the trend, ARIMA can also model the trend.


#?#############################################################
#? Hyperparameter Optimization (Determining model degrees)
#?#############################################################

#!############################################################
#! Determining Model Degree According to AIC & BIC Statistics
#!############################################################

#? AIC (Akaike Information Criterion) -> the lower the better

p = d = q = range(0,4)
pdq = list(itertools.product(p,d,q))

def arima_optimizer(train, orders):
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

best_params_aic = arima_optimizer(train, pdq)

#?#####################
#? Final Model
#?#######################

arima_model = ARIMA(train, order=best_params_aic).fit()
y_pred = arima_model.forecast(steps=48)
y_pred = pd.Series(y_pred, index = test.index)


plot_co2(train, test, y_pred, "ARIMA")



#?############################################################
#? SARIMA (p,d,q) (P,D,Q)m
#? Seasonal AutoRegressive Integrated Moving Average
#?############################################################
"""
 (p,d,q)-> ARIMA parameters
(P,D,Q) -> SARIMA parameters
 m -> Seasonality Parameter

SARIMA = ARIMA + Seasonality
Trend ve/veya mevsimsellik içeren tek değişkenli serilerde kullanılabilir.
Diğer istatistiksel metotlara göre en gelişmiş metottur.
p,d,q ARIMA'dan gelen parametrelerdir. #!Trend elemanlarıdır.
ARIMA trendi modelleyebilir.

p: gerçek değer gecikme sayıısı (oto regresif derece) 
p=2 ise yt-1 ve yt-2 modeldedir.

d: fark işlemi sayısı (fark derecesi, I) -> durağanlaştırma için.

q: hata gecikme sayısı (hareketli ortalama derecesi)
q=2 ise et-1 ve et-2 modeldedir.

P,D,Q mevsimsel gecikme sayularıdır. #! Season elemanlarıdır.

m: tek bir mevsimlik dönem için zaman adımı sayısı. Mevsimselliğin görülme yapısını ifade eder.


Durağan: SES,AR, MA, ARMA
Trend: DES, ARIMA
Trend ve Mevsimsellik: TES, SARIMA
"""


#?##########################################################################
#? SARIMA MODEL (Seasonal AutoRegressive Integrated Moving Average)
#?##########################################################################

sarima_model = SARIMAX(train, order = (1,1,1), seasonal_order = (0,0,0,12))
sarima_model = sarima_model.fit()

y_pred_test = sarima_model.get_forecast(steps=48)

y_pred = y_pred_test.predicted_mean

y_pred = pd.Series(y_pred, index = test.index)

plot_co2(train, test, y_pred, "SARIMA")

#?#############################################################
#? Hyperparameter Optimization (Determining model degrees)
#?#############################################################
p = d = q = range(0,2)

pdq = list(itertools.product(p,d,q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p,d,q))]

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

best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)


#?#####################
#? Final Model
#?#######################

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit()

y_pred_test = sarima_final_model.get_forecast(steps=48)

y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index = test.index)

plot_co2(train, test, y_pred, "SARIMA")