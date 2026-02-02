#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price column
df = df.drop(columns=['Weighted_Price'], errors='ignore')

# Rename Timestamp -> Date and convert to datetime
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index on Date
df = df.set_index('Date')

# Fill missing values
df['Close'] = df['Close'].fillna(method='ffill')

for col in ('High', 'Low', 'Open'):
    df[col] = df[col].fillna(df['Close'])

df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Keep 2017+ and resample daily with required aggregations
df = df.loc['2017-01-01':]

df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Return the transformed df before plotting (in a script: keep it as df)
print(df)

df.plot()
plt.savefig('14visualizepandas.png')
plt.show()
