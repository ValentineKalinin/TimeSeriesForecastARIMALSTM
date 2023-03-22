import pandas as pd

df = pd.read_csv('Datasets/LTC-USD_5min.csv', converters={'<TIME>': '{:0>6}'.format})
print(df.head())

df['<DATE>'] = df['<DATE>'].astype(str) + ' ' + df['<TIME>'].astype(str)
del df['<TIME>']
del df['<VOL>']
del df['<HIGH>']
del df['<LOW>']
df['<DATE>'] = pd.to_datetime(df['<DATE>'], format='%Y-%m-%d %H:%M:%S')
df.to_csv('Datasets/LTC-USD_5min_2.csv', index=False)
print(df.head())
