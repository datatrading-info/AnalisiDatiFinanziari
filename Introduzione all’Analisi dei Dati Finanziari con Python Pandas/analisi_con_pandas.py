import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import yfinance as yf


start_date = '2010-01-01'
end_date = '2020-01-01'
AAPL = yf.download('AAPL', start_date, end_date)
GOOG = yf.download('GOOG', start_date, end_date)
MSFT = yf.download('MSFT', start_date, end_date)
AMZN = yf.download('AMZN', start_date, end_date)

print(AAPL.head())

AAPL['Adj Close'].plot()

AAPL['Volume'].plot()

AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
AAPL_mean = AAPL['Daily Return'].mean()

print(AAPL_mean)

tech_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
closing_df = yf.download(tech_list, start_date, end_date)['Adj Close']

rets = closing_df.pct_change()

sns.jointplot(x='GOOG', y='MSFT', data=rets, kind='scatter')
