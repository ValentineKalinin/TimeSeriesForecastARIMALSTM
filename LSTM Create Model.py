# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# импортируем библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# загружаем данные о цене криптовалюты (Bitcoin)
df = pd.read_csv('Datasets/LTC-USD_5min_4.csv', parse_dates=['<DATE>'], index_col='<DATE>')
# print(el_df.head())
# df = el_df.resample('12H').mean()
# отображаем график цены
plt.figure(figsize=(10, 6))
plt.plot(df['<CLOSE>'])
plt.title('Litecoin Price')
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()

# разбиваем данные на обучающую и тестовую выборки
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size - 1], df.iloc[train_size:len(df)]
# print('df.iloc[0:train_size]', df.iloc[train_size:len(df)], '\n')
# print('train ', len(train), 'test ', len(test))
# print('train ', train[:5])
# print('test ', test[:5])

# нормализуем данные в диапазон от 0 до 1
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)


# print('train_scaled ', train_scaled, len(train_scaled))
# print('test_scaled ', test_scaled, len(test_scaled))


# создаем функцию для преобразования данных в формат X и y для LSTM
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)


# определяем размер окна (количество предыдущих наблюдений для прогноза)
window_size = 10

# преобразуем обучающую и тестовую выборки в формат X и y для LSTM
X_train, y_train = create_dataset(train_scaled, window_size)
X_test, y_test = create_dataset(test_scaled, window_size)
# print('X_train, y_train \n', X_train[:5], len(X_train), '\n', y_train[:5], len(y_train))
# print('X_test, y_test \n', X_test[:5], len(X_test), '\n', y_test[:5], len(y_test))

# меняем размерность данных для подачи в LSTM (количество образцов, количество шагов времени, количество признаков)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# print('X_train, X_test ', len(X_train), ' ', len(X_test))

# создаем модель LSTM с четырьмя слоями (входным, двумя скрытыми и выходным)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(50))
model.add(Dense(1))
print('model ', model.summary())

# компилируем модель с функцией потерь MSE и оптимизатором Adam
model.compile(loss='mean_squared_error', optimizer='adam')

# обучаем модель на обучающей выборке с количеством эпох 100 и размером батча 32
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.05, verbose=1,
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])

# делаем прогноз на тестовой выборке
y_pred = model.predict(X_test)
# print('y_pred \n', y_pred[:6], len(y_pred))
# print('y_test \n', y_test[:6], len(y_test))

# обратно преобразуем данные в исходный масштаб:
# создаем пустую таблицу с 4 полями, помещаем наш прогноз в правое поле, инвертируем
y_pred_dataset_like = np.zeros(shape=(len(y_pred), 4))
y_pred_dataset_like[:, 0] = y_pred[:, 0]
y_pred = scaler.inverse_transform(y_pred_dataset_like)[:, 0]
# print('y_pred \n', y_pred[:6], len(y_pred))

# print('y_test.reshape(-1, 1) \n', y_test.reshape(-1, 1))
# y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_test = y_test.reshape(-1, 1)
y_test_dataset_like = np.zeros(shape=(len(y_test), 4))
y_test_dataset_like[:, 0] = y_test[:, 0]
y_test = scaler.inverse_transform(y_test_dataset_like)[:, 0]
# print('y_test \n', y_test[:6], len(y_test))

# вычисляем среднеквадратическую ошибку прогноза
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print('RMSE: ', rmse)

# отображаем график фактической и прогнозной цены
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Litecoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Генерируем описание модели в формате json
model_json = model.to_json()
# Записываем модель в файл
json_file = open("Models/lstm_model_4.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("Models/lstm_model_4.h5")
print("Сохранили Model")
