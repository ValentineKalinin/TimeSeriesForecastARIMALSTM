# Загружаем Model
from keras.models import model_from_json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# загружаем данные о цене криптовалюты (Bitcoin)
el_df = pd.read_csv('Datasets/LTC-USD_5min_4.csv', parse_dates=['<DATE>'], index_col='<DATE>')
print(el_df.head())
df = el_df.resample('d').mean()
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
print('train ', len(train), ' test ', len(test))

# нормализуем данные в диапазон от 0 до 1
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)


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

# меняем размерность данных для подачи в LSTM (количество образцов, количество шагов времени, количество признаков)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Загружаем данные об архитектуре сети из файла json
json_file = open("Models/lstm_model_4.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель на основе загруженных данных
loaded_model = model_from_json(loaded_model_json)
# Загружаем веса в модель
loaded_model.load_weights("Models/lstm_model_4.h5")

# компилируем модель с функцией потерь MSE и оптимизатором Adam
loaded_model.compile(loss='mean_squared_error', optimizer='adam')

# делаем прогноз на тестовой выборке
y_pred = loaded_model.predict(X_test)

# обратно преобразуем данные в исходный масштаб:
# создаем пустую таблицу с 4 полями, помещаем наш прогноз в правое поле, инвертируем
y_pred_dataset_like = np.zeros(shape=(len(y_pred), 4))
y_pred_dataset_like[:, 0] = y_pred[:, 0]
y_pred = scaler.inverse_transform(y_pred_dataset_like)[:, 0]

y_test = y_test.reshape(-1, 1)
y_test_dataset_like = np.zeros(shape=(len(y_test), 4))
y_test_dataset_like[:, 0] = y_test[:, 0]
y_test = scaler.inverse_transform(y_test_dataset_like)[:, 0]

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

