import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import pmdarima as pm

# 1. Open CSV with data
# 2x -> LTC-USD_5min_2.csv
# 4x -> LTC-USD_5min_4.csv
df = pd.read_csv('Datasets/LTC-USD_5min_4.csv')
df['<DATE>'] = pd.to_datetime(df['<DATE>'])
# print(df.head())


# 2. Draw basic plot of all range
fig = px.line(df, x='<DATE>', y='<CLOSE>', title='LTC/USDT')
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(step="all")
        ])
    )
)
# fig.show()


el_df = df.set_index('<DATE>')
# el_df.plot(subplots=True)
fig, ax = plt.subplots()
plt.title('Basic Plot')
ax.plot(el_df)
plt.show()


# # 3. Replace all zero values
# print("\nMissing values :  ", df.isnull().any())
# df['<OPEN>'] = df['<OPEN>'].fillna(method='ffill')
# df['<CLOSE>'] = df['<CLOSE>'].fillna(method='ffill')
# df['<HIGH>'] = df['<HIGH>'].fillna(method='ffill')
# df['<LOW>'] = df['<LOW>'].fillna(method='ffill')
# print("\nMissing values :  ", df.isnull().any())


# 4. Group values for better prediction
final_df = el_df.resample('d').mean()
# el_df.resample('d').mean().plot(subplots=True)
fig, ax = plt.subplots()
ax.plot(final_df)
plt.title('ZIP Plot')
plt.show()
print(final_df.head())


# 5. Auto ARIMA model -> (p,d,q)
model = pm.auto_arima(final_df['<CLOSE>'],
                      # m=12, seasonal=True,
                      start_p=1, statr_d=1, start_q=1,
                      max_p=7, max_d=3, max_q=3,
                      test='adf', error_action='ignore',
                      suppress_warnings=True, trace=True, stepwise=False)


# 6. Split for Train and Test range
split = int(len(final_df) - 4)
# split = int(0.8 * len(final_df))
train = final_df[:split - 1]
test = final_df[split:]
n_per = len(final_df) - split
print('split ', split, ' prediction ', n_per)
# train = df[:1491]
# # (df.index.get_level_values(0) >= '2012-01-31') & (df.index.get_level_values(0) <= '2017-04-30')]
# # test = df[(df.index.get_level_values(0) > '2017-04-30')]
# test = df[1492:1507]
print('test ', test)


# 7. Test prediction
model.fit(train['<CLOSE>'])
forecast = model.predict(n_periods=4, return_conf_int=True)
print('forecast ', forecast)
forecast_df = pd.DataFrame(forecast[0], index=test.index, columns=['Prediction'])
print('forecast_df ', forecast_df)

# pd.concat([final_df['<CLOSE>'][split - 1:], forecast_df], axis=1).plot()
fig, ax = plt.subplots()
ax.plot(pd.concat([final_df['<CLOSE>'][split - 1:], forecast_df], axis=1))
plt.title('Forecast Test')
plt.show()


# 8. Future prediction
forecast_future = model.predict(n_periods=4, return_conf_int=True)
print('forecast_future ', forecast_future)
forecast_range = pd.date_range(start='2022-04-30', periods=4, freq='d')
print('forecast_range ', forecast_range)
forecast_future_df = pd.DataFrame(forecast_future[0], index=forecast_range, columns=['Prediction'])
print('forecast_future_df ', forecast_future_df)

# pd.concat([final_df['<CLOSE>'][split - 1:], forecast_future_df], axis=1).plot()
fig, ax = plt.subplots()
plt.title('Forecast Future')
ax.plot(pd.concat([final_df['<CLOSE>'][split - 1:], forecast_future_df], axis=1))
plt.show()

fig = px.line(pd.concat([final_df['<CLOSE>'][split - 1:], forecast_future_df], axis=1), x='<DATE>', y='<CLOSE>', title='LTC/USDT')
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(step="all")
        ])
    )
)
fig.show()
