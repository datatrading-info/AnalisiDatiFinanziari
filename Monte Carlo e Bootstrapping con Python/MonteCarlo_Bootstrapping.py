import pandas as pd
import numpy as np
from functools import reduce
import yfinance as yf
import datetime
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.style.use('ggplot')
figsize = (15, 8)

start, end = '2010-01-01', '2020-01-01'
tickers = ["^DJI", "^IXIC", "^GSPC", "^STOXX50E", "^N225", "^GDAXI"]
asset_universe = pd.DataFrame([yf.download(ticker, start, end).loc[:, 'Adj Close'] for ticker in tickers],
                              index=tickers)
asset_universe = asset_universe.T.fillna(method='ffill')
asset_universe = asset_universe/asset_universe.iloc[0, :]

asset_universe.plot(figsize=figsize)
plt.show()

portfolio_returns = asset_universe.pct_change().dropna().mean(axis=1)
portfolio = (asset_universe.pct_change().dropna().mean(axis=1) + 1).cumprod()
asset_universe.plot(figsize=figsize, alpha=0.4)
portfolio.plot(label='Portfolio', color='black')
plt.legend()
plt.show()

portfolio_bootstrapping = pd.DataFrame([random.choices(list(portfolio_returns.values), k=252) for i in range(1000)])
portfolio_bootstrapping = (1+portfolio_bootstrapping.T.shift(1).fillna(0)).cumprod()
portfolio_bootstrapping.plot(figsize=figsize, legend=False, linewidth=1, alpha=0.2, color='b')
plt.show()

asset_universe_returns = asset_universe.pct_change()
portfolio_constituents_bootstrapping = pd.DataFrame([((asset_universe_returns.iloc[random.choices(
    range(len(asset_universe)), k=252)]).mean(axis=1)+1).cumprod().values
    for x in range(1000)]).T

portfolio_constituents_bootstrapping.plot(figsize=figsize, legend=False, linewidth=1, alpha=0.2, color='purple')
plt.show()

mu = portfolio_returns.mean()
sigma = portfolio_returns.std()

print(f'Our portfolio mean return value is {round(mu*100,2)}%')
print(f'Our portfolio standard deviation value is {round(sigma*100,2)}%')

portfolio_mc = pd.DataFrame([(np.random.normal(loc=mu, scale=sigma, size=252)+1) for x in range(1000)]).T.cumprod()
portfolio_mc.plot(figsize=figsize, legend=False, linewidth=1, alpha=0.2, color='green')
plt.show()

for asset in (asset_universe_returns.mean() * 100).round(2).index:
    print(f'The mean return for {asset} is {(asset_universe_returns.mean() * 100).round(2)[asset]}%')

print('\n')
for asset in (asset_universe_returns.std() * 100).round(2).index:
    print(f'The standard deviation for {asset} is {(asset_universe_returns.std() * 100).round(2)[asset]}%')

asset_returns_dfs = []
for asset in asset_universe_returns.mean().index:
    mu = asset_universe_returns.mean()[asset]
    sigma = asset_universe_returns.std()[asset]
    asset_mc_rets = pd.DataFrame([(np.random.normal(loc=mu, scale=sigma, size=252)) for x in range(1000)]).T
    asset_returns_dfs.append(asset_mc_rets)

weighted_asset_returns_dfs = [(returns_df / len(tickers)) for returns_df in asset_returns_dfs]

portfolio_constituents_mc = (reduce(lambda x, y: x + y, weighted_asset_returns_dfs) + 1).cumprod()

portfolio_constituents_mc.plot(figsize=figsize, legend=False, linewidth=1, alpha=0.2, color='orange')
plt.show()

ax, fig = plt.subplots(figsize=(12,10))
sns.heatmap(asset_universe_returns.corr(),annot=True)
plt.plot()
plt.show()
print("")