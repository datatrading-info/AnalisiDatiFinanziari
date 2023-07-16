import pandas as pd
import numpy as np
import ffn
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

num_days = 1000
data = (np.random.randn(num_days) + np.random.uniform(low=0.0, high=0.2, size=num_days))
index = pd.date_range('01/01/2010', periods=num_days, freq='D')
data = pd.DataFrame(data, index=index, columns=['Returns'])
data['Equity'] = data.cumsum() + 100
data.iloc[0] = 100

perf = data['Equity'].calc_stats()

print(type(perf))

perf.plot()

perf.display()

perf.display_monthly_returns()

ffn.to_drawdown_series(data['Equity']).plot(figsize=(15,7),grid=True)

perf.plot_histogram()

print(perf.stats)

print(perf.stats['yearly_sharpe'])

print(perf.display_lookback_returns())