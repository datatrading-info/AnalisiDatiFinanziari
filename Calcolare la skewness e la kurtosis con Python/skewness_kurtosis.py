import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

GOOG = yf.download('GOOG', start='2000-01-01', end='2020-01-01')
print(GOOG.head())

GOOG['Percentage Returns'] = GOOG['Adj Close'].pct_change()
print(GOOG.head())

GOOG['Percentage Returns'].plot(kind='hist',bins=100)

print('Mean =:',GOOG['Percentage Returns'].mean())
print('Variance =:',GOOG['Percentage Returns'].var())

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Converte il dataframe pandas in un array numpy array e lo ordina
h = np.asarray(GOOG['Percentage Returns'].dropna())
h = sorted(h)

# usa il modulo stats di scipy per allenare una distribuzione normale con stessa media e deviazione standard
fit = stats.norm.pdf(h, np.mean(h), np.std(h))

# Grafico dell'istogramma di entrambe le serie
plt.plot(h, fit, '-', linewidth=2)
plt.hist(h, normed=True, bins=100)
plt.show()

print('Skew =', GOOG['Percentage Returns'].skew())
print('Kurtosis =', GOOG['Percentage Returns'].kurt())
print("")