# Time Series Analysis and Forcasting with Facebook Kats

Example code files and referenced papers presented on YAZSUM 2021 AI Event

## Kats Info

Kats is a a lightweight, easy-to-use, and generalizable time series analysis framework from Facebook. It performs various time series analysis taks, including detection, forecasting, feature extraction.

Kats is available for download on [PyPI](https://pypi.python.org/pypi/kats/) with just "pip install kats".

## Important links

- Kats Homepage: https://facebookresearch.github.io/Kats/
- Kats Python package: https://pypi.org/project/kats/0.1.0/
- Source code repository: https://github.com/facebookresearch/kats
- Official Tutorials: https://github.com/facebookresearch/Kats/tree/master/tutorials
- Video Tutorial: https://youtu.be/GzP2xdoVqJE (in Turkish with English subtitles)

## Contents:

### Forecasting

- [sarima.py](sarima.py)	Basic Forecasting example with SARIMA Model
- [prophet.py](prophet.py)	Basic Forecasting example with FB Prophet Model
- [theta.py](theta.py)	Basic Forecasting example with Theta Model
- [holtwinters.py](holtwinters.py)    Basic Forecasting example with Holt-Winters Model
- [ensemble.py](ensemble.py)	      Basic Forecasting example with an Ensemble Model of ARIMA, Prophet, Theta, and Linear Models

### Hyperparameter Tuning and Backtesting

- [arima.py](arima.py)	  Hyper parameter tuning an ARIMA model with Grid Search method
- [backtest.py](backtest.py)	Backtesting an ARIMA and a Prophet Model using metrics mape, smape, mae, mase, mse, rmse

### Changepoint, Outlier, and Trend Detection

- [cusum.py](cusum.py)	  Changepoint detection with CUSUM method
- [bocpd.py](bocpd.py)	  Changepoint detection with Bayesian Online Change Point Detection (BOCPD) method
- [outlier.py](outlier.py)	      Outlier detection and removal with and without interpolation
- [trend.py](trend.py)		      Trend detection

### Time Series Features

- [features.py](features.py)	Extracting statistical time series features

### Papers
- [Bayesian Online Changepoint Detection](0710.3742.pdf)
- [Self-supervised learning for fast and scalable time series hyper-parameter tuning](2102.05740.pdf)

