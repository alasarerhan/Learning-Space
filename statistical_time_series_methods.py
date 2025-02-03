

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

