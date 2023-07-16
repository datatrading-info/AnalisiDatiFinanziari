import pandas as pd
import numpy as np


np.random.seed(1)
num_days = 1000
index = pd.date_range('01/01/2010', periods=num_days, freq='D')
data1 = pd.DataFrame((np.random.randn(num_days) + np.random.uniform(low=0.0, high=0.2, size=num_days)),index=index,columns=['Data1'])
data2 = pd.DataFrame((np.random.randn(num_days) + np.random.uniform(low=0.0, high=0.2, size=num_days)),index=index,columns=['Data2'])
data3 = pd.DataFrame((np.random.randn(num_days) + np.random.uniform(low=0.0, high=0.2, size=num_days)),index=index,columns=['Data3'])
data4 = pd.DataFrame((np.random.randn(num_days) + np.random.uniform(low=0.0, high=0.2, size=num_days)),index=index,columns=['Data4'])

data = pd.concat([data1,data2,data3,data4],axis=1)
data = data.cumsum() + 100
data.iloc[0] = 100
print(data.head())

import ffn

perf = data.calc_stats()

print(type(perf))

perf.plot()

import yfinance as yf

stocks = ['AAPL', 'AMZN', 'MSFT', 'NFLX']

data = yf.download(stocks, start='2010-01-01', end='2020-01-01')['Adj Close']
data.sort_index(ascending=True, inplace=True)
perf = data.calc_stats()

perf.plot()

perf.display()

print(perf.stats)

print(perf.stats.loc['cagr'])

returns = data.to_log_returns().dropna()
print(returns.head())

ax = returns.hist(figsize=(20, 10),bins=30)

print(returns.corr().as_format('.2f'))

returns.plot_corr_heatmap()

print(returns.calc_mean_var_weights().as_format('.2%'))

print(perf.display_lookback_returns().loc['mtd'])

perf.plot_scatter_matrix()

ffn.to_drawdown_series(data).plot(figsize=(15,10))
print("")