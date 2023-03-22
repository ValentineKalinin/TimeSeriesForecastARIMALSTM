import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Importing data
df = pd.read_csv('Datasets/LTC-USD_5min.csv', converters={'<TIME>': '{:0>6}'.format})
print(df)
# df_all = (df['<OPEN>']+df['<CLOSE>']+df['<HIGH>']+df['<LOW>'])/4
df_all = (df['<OPEN>']+df['<CLOSE>'])/2

# Adfuller test (проверка на стационарность)
res0 = adfuller(df['<CLOSE>'].dropna())
print('\nAugmented Dickey-Fuller Statistic: %f' % res0[0])
print('p-value: %f' % res0[1])
res1 = adfuller(df['<CLOSE>'].diff().dropna())
print('\nAugmented Dickey-Fuller Statistic: %f' % res1[0])
print('p-value: %f' % res1[1])
res2 = adfuller(df['<CLOSE>'].diff().diff().dropna())
print('\nAugmented Dickey-Fuller Statistic: %f' % res2[0])
print('p-value: %f' % res2[1])

# The Genuine Series
plt.rcParams.update({'figure.figsize': (26, 9), 'figure.dpi': 60})

fig, axes = plt.subplots(1, 1, sharex=True)
plt.xlabel('Index', fontsize=20)
plt.ylabel('Value', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
axes.grid(True)
axes.plot(df_all.dropna())
axes.set_title('The Genuine Series')
plt.show()
plt.style.use('seaborn-darkgrid')

# Graphics
fig, axes = plt.subplots(3, sharex=True)
# The Genuine Series
axes[0].plot(df_all)
axes[0].set_title('The Genuine Series')
# Order of Differencing: First
axes[1].plot(df_all.diff(1))
axes[1].set_title('Order of Differencing: First')
# Order of Differencing: Second
axes[2].plot(df_all.diff(2))
axes[2].set_title('Order of Differencing: Second')
plt.show()

# Auto Correlation
fig, axes = plt.subplots(4, sharex=True)
plot_acf(df_all, lags=30, ax=axes[0])
plot_acf(df_all.diff(1).dropna(), lags=10, ax=axes[1])
plot_acf(df_all.diff(2).dropna(), lags=10, ax=axes[2])
plot_acf(df_all.diff(3).dropna(), lags=10, ax=axes[3])
plt.show()

# Partial Auto Correlation
fig, axes = plt.subplots(4, sharex=True)
plot_pacf(df_all, lags=30, ax=axes[0], method="ywm")
plot_pacf(df_all.diff(1).dropna(), lags=10, ax=axes[1], method="ywm")
plot_pacf(df_all.diff(2).dropna(), lags=10, ax=axes[2], method="ywm")
plot_pacf(df_all.diff(3).dropna(), lags=10, ax=axes[3], method="ywm")
plt.show()


# Build Model
mymodel = ARIMA(df, order=(3, 3, 2))
modelfit = mymodel.fit()
print(modelfit.summary())

# Plot residual errors
residuals = pd.DataFrame(modelfit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Create Training and Test
split = int(0.8 * len(df_all))
train = df_all[:split - 1]
test = df_all[split:]
print("Train: \n", train)
print("Test: \n", test)
# train = df['<CLOSE>'][120:201]
# test = df['<CLOSE>'][200:215]
# print("Train: \n", train)
# print("Test: \n", test)

plt.plot(train)
plt.plot(test)
plot_acf(train.diff(1).dropna(), lags=15)
plot_pacf(train.diff(1).dropna(), lags=15, method="ywm")

# Build Model
model = ARIMA(train, order=(1, 0, 1))
fitted = model.fit()
print(fitted.summary())

# Build Model
model = ARIMA(train, order=(1, 0, 2))
fitted = model.fit()
print(fitted.summary())

# # Actual vs Fitted
# fitted.predict().plot()
# plt.show()

# Make as pandas series
fcast = fitted.get_forecast(steps=41504).summary_frame(alpha=0.05)
print('Fcast ', fcast[0:10], fcast[len(fcast)-10:len(fcast)])
fcast.to_csv('Datasets/fcast.csv', index=True, index_label='ID')
fc_series = pd.Series(fcast['mean'], index=test.index)
lower_series = pd.Series(fcast['mean_ci_lower'], index=test.index)
upper_series = pd.Series(fcast['mean_ci_upper'], index=test.index)
# Evaluation Metric
print("\n MAE : \n ", mean_absolute_error(fcast['mean'], test))
print("\n RMSLE : \n", mean_squared_log_error(fcast['mean'], test))
print("\n MAPE : \n", mean_absolute_percentage_error(fcast['mean'], test))

# Plot
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=0.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Get prediction for test duration
predictions = pd.Series(fitted.forecast(len(test)))
predictions = predictions.map(lambda x: x if x >= 0 else 0)
print('Pred Forecast', predictions[:40])
plt.plot(predictions, label='predictions')
actuals = test
# Evaluation Metric
print("\n MAE : \n ", mean_absolute_error(predictions, actuals))
print("\n RMSLE : \n", mean_squared_log_error(predictions, actuals))
print("\n MAPE : \n", mean_absolute_percentage_error(predictions, actuals))

# df = pd.read_csv('Datasets/LTC-USD_5min.csv', converters={'<TIME>': '{:0>6}'.format})
# df['<TIME>'].astype(str)
# df.to_csv('Datasets/LTC-USD_5min.csv', index=False)
# print(df.head())


# df['<DATE>'] = pd.to_datetime(df['<DATE>'])
# print(df.head())
# df.info()

# ARIMA

# df2 = pd.read_csv('Datasets/LTC-USD_5min.csv', converters={'<TIME>': '{:0>6}'.format})
#
# # Adfuller test
# res = adfuller(df['<CLOSE>'].diff().dropna())
# print('\nAugmented Dickey-Fuller Statistic: %f' % res[0])
# print('p-value: %f' % res[1])
#
# # plt.rcParams.update({'figure.figsize': (40, 9), 'figure.dpi': 120})
# fig, axes = plt.subplots(1, 1, sharex=True)
# # The Genuine Series
# axes.plot(df2['<CLOSE>'].dropna())
# axes.set_title('The Genuine Series')
# plt.show()
#
# fig, axes = plt.subplots(6, 2, sharex=True)
# # The Genuine Series
# axes[0, 0].plot(df['<CLOSE>'])
# axes[0, 0].set_title('The Genuine Series')
# plot_acf(df['<CLOSE>'], ax=axes[0, 1])
# plot_pacf(df['<CLOSE>'], ax=axes[1, 1], method="ywm")
# # Order of Differencing: First
# axes[2, 0].plot(df['<CLOSE>'].diff())
# axes[2, 0].set_title('Order of Differencing: First')
# plot_acf(df['<CLOSE>'].diff().dropna(), ax=axes[2, 1])
# plot_pacf(df['<CLOSE>'].diff().dropna(), ax=axes[3, 1], method="ywm")
# # Order of Differencing: Second
# axes[4, 0].plot(df['<CLOSE>'].diff().diff())
# axes[4, 0].set_title('Order of Differencing: Second')
# plot_acf(df['<CLOSE>'].diff().diff().dropna(), ax=axes[4, 1])
# plot_pacf(df['<CLOSE>'].diff().diff().dropna(), ax=axes[5, 1], method="ywm")
# plt.show()
#
#
# # Build Model
# mymodel = ARIMA(df['<CLOSE>'], order=(2, 1, 1))
# modelfit = mymodel.fit()
# print(modelfit.summary())
#
# # Forecast
# fc, se, conf = modelfit.forecast(15, alpha=0.05)  # 95% conf
#
# # Create Training and Test
# train = df.value[:85]
# test = df.value[15:]
#
# # Make as pandas series
# fc_series = pd.Series(fc, index=test.index)
# lower_series = pd.Series(conf[:, 0], index=test.index)
# upper_series = pd.Series(conf[:, 1], index=test.index)
#
# # Plot
# plt.figure(figsize=(12, 5), dpi=100)
# plt.plot(train, label='training')
# plt.plot(test, label='actual')
# plt.plot(fc_series, label='forecast')
# plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
# plt.title('Forecast vs Actuals')
# plt.legend(loc='upper left', fontsize=8)
# plt.show()
#
# Plotting Residual Errors
# myresiduals = pd.DataFrame(fitted.resid)
# fig, ax = plt.subplots(1, 2)
# myresiduals.plot(title="Residuals", ax=ax[0])
# myresiduals.plot(kind='kde', title='Density', ax=ax[1])
# plt.show()


# # # SARIMA
# # Plot
# fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=100, sharex=True)
#
# # Usual Differencing
# axes[0].plot(data['<CLOSE>'], label='Original Series')
# axes[0].plot(data['<CLOSE>'].diff(1), label='Usual Differencing')
# axes[0].set_title('Usual Differencing')
# axes[0].legend(loc='upper left', fontsize=10)
#
#
# # Seasinal Dei
# axes[1].plot(data['<CLOSE>'], label='Original Series')
# axes[1].plot(data['<CLOSE>'].diff(12), label='Seasonal Differencing', color='green')
# axes[1].set_title('Seasonal Differencing')
# plt.legend(loc='upper left', fontsize=10)
# plt.suptitle('Plot LTC/USDT', fontsize=16)
# plt.show()
#
# # !pip3 install pyramid-arima
# import pmdarima as pm
#
# # Seasonal - fit stepwise auto-ARIMA
# smodel = pm.auto_arima(data['<CLOSE>'],
#                        start_p=1, start_q=1, test='adf',
#                        max_p=2, max_q=2, d=1, m=2,
#                        start_P=0, start_Q=0, D=1, seasonal=True,
#                        max_P=2, max_D=2,
#                        trace=True, error_action='ignore',
#                        suppress_warnings=True, stepwise=True)
#
# smodel.summary()
#
# # Forecast
# n_periods = 2
# fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
# index_of_fc = pd.date_range(data.index[-1], periods=n_periods, freq='MS')
#
# # make series for plotting purpose
# fitted_series = pd.Series(fitted, index=index_of_fc)
# lower_series = pd.Series(confint[:, 0], index=index_of_fc)
# upper_series = pd.Series(confint[:, 1], index=index_of_fc)
#
# # Plot
# plt.plot(data)
# plt.plot(fitted_series, color='darkgreen')
# plt.fill_between(lower_series.index,
#                  lower_series,
#                  upper_series,
#                  color='k', alpha=.15)
#
# plt.title("SARIMA - Final Forecast of a10 - Drug Sales")
# plt.show()
