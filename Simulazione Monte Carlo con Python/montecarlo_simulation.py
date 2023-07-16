import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from scipy.stats import norm
import yfinance as yf

# Scarica i dati dei prezzi di Apple in un dataframe
apple = yf.download('AAPL', start='2000-01-01', end='2020-01-01')

# Calcolo del CARG che è una stima dei redimenti medi (mu)
days = (apple.index[-1] - apple.index[0]).days
cagr = ((((apple['Adj Close'][-1]) / apple['Adj Close'][1])) ** (365.0 / days)) - 1
print('CAGR =', str(round(cagr, 4) * 100) + "%")

# crea una serie di rendimenti percentuali e calcola la volatilità annuale dei rendimenti
apple['Returns'] = apple['Adj Close'].pct_change()
volatility = apple['Returns'].std() * sqrt(252)
print("Annual Volatility =", str(round(volatility, 4) * 100) + "%")

np.random.seed(1)
# Definizione delle Variabili
S = apple['Adj Close'][-1] # prezzo iniziale dell'asset (es. l'ultimo prezzo reale)
T = 252 # Numero di giorni di trading
mu = cagr # Redimenti medi
vol = volatility # Volatilità

# crea una lista di rendimenti giornalieri usando una distrubuzione normale casuale
daily_returns=np.random.normal((mu/T),vol/sqrt(T),T)+1

# imposta il prezzo iniziale e care una serie di prezzi a partire dai rendimenti giornalieri casuali
price_list = [S]
for x in daily_returns:
    price_list.append(price_list[-1]*x)

# Crea i gradici della serie dei prezzi e dell'istogramma dei rendimenti giornalieri
plt.plot(price_list)
plt.show()
plt.hist(daily_returns-1, 100)
plt.show()

