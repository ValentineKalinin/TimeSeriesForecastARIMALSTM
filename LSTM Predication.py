# импортируем библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

# загружаем данные о цене криптовалюты (Bitcoin)
el_df = pd.read_csv('Datasets/LTC-USD_5min_4.csv', parse_dates=['<DATE>'], index_col='<DATE>')
print(el_df.head())
df = el_df.resample('4H').mean()
# отображаем график цены
plt.figure(figsize=(10, 6))
plt.plot(df['<CLOSE>'])
plt.title('Bitcoin Price')
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()

# разбиваем данные на обучающую и тестовую выборки
train_size = int(len(df) * 0.98)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))

# нормализуем данные в диапазон от 0 до 1
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)


# создаем функцию для преобразования данных в формат X и y для LSTM
def create_dataset(data, window_size, forecast_size):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_size):
        X.append(data[i:(i + window_size), 0])
        y.append(data[(i + window_size):(i + window_size + forecast_size), 0])
    return np.array(X), np.array(y)


# определяем размер окна (количество предыдущих наблюдений для прогноза)
window_size = 10

# определяем размер прогноза (количество будущих наблюдений для прогноза)
forecast_size = 10

# преобразуем обучающую и тестовую выборки в формат X и y для LSTM
X_train, y_train = create_dataset(train_scaled, window_size, forecast_size)
X_test, y_test = create_dataset(test_scaled, window_size, forecast_size)

# меняем размерность данных для подачи в LSTM (количество образцов, количество шагов времени, количество признаков)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# создаем модель LSTM с четырьмя слоями (входным, двумя скрытыми и выходным)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(50))
model.add(Dense(forecast_size))  # выходной слой с размером равным количеству шагов прогноза
model.add(Activation('linear'))  # линейная функция активации для регрессии

# компилируем модель с функцией потерь MSE и оптимизатором Adam
model.compile(loss='mse', optimizer='adam')

# обучаем модель на обучающей выборке с количеством эпох 100 и размером батча 32
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.05, verbose=2,
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])

# делаем прогноз на тестовой выборке
y_pred = model.predict(X_test)
print(y_pred[:10], y_pred.shape)

# y_pred_dataset_like = np.zeros(shape=(len(y_pred), 2))
# y_pred_dataset_like[:, 0] = y_pred[:, 0]
# y_pred = scaler.inverse_transform(y_pred_dataset_like)[:, 0]
# print(y_pred[:10], y_pred.shape)
#
# y_test_dataset_like = np.zeros(shape=(len(y_test), 2))
# y_test_dataset_like[:, 0] = y_test[:, 0]
# y_test = scaler.inverse_transform(y_test_dataset_like)[:, 0]
# print(y_test[:10], y_test.shape)
# создаем новый scaler для столбца Close
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_close.fit(train[['<CLOSE>']])
# обратно преобразуем данные в исходный масштаб
y_test = scaler_close.inverse_transform(y_test)
y_pred = scaler_close.inverse_transform(y_pred)
# вычисляем среднеквадратическую ошибку прогноза по каждому шагу
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=0))
print('RMSE: ', rmse)

for i in range(forecast_size):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, i], label='Actual')
    plt.plot(y_pred[:, i], label='Predicted')
    plt.title('Bitcoin Price Prediction - {} days ahead'.format(i + 1))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
