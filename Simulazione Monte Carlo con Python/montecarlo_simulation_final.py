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


# lista vuota per i valori finali di ogni simulazione
result = []

# Definizione delle Variabili
S = apple['Adj Close'][-1] # prezzo iniziale dell'asset (es. l'ultimo prezzo reale)
T = 252 # Numero di giorni di trading
mu = cagr # Redimenti medi
vol = volatility # Volatilità


# Specifica il numero di simulazioni
num_sims = 1000
for i in range(num_sims):
    # crea una lista di rendimenti giornalieri usando una distrubuzione normale casuale
    daily_returns = np.random.normal(mu / T, vol / sqrt(T), T) + 1

    # imposta il prezzo iniziale e care una serie di prezzi a partire dai rendimenti giornalieri casuali
    price_list = [S]

    for x in daily_returns:
        price_list.append(price_list[-1] * x)
    # grafico dei dati di un ogni singola simulazione che saranno visualizzate al termine
    plt.plot(price_list)

    # aggiunta del valore finale di ogni simulazione alla lista dei risultati
    result.append(price_list[-1])

# visualizzazione del grafico delle serie dei prezzi
plt.show()
# creazione dell'istrogramma dei prezzi finali di ogni simulazione
plt.hist(result, bins=50)
plt.show()

# uso della funzione mean di numpy per calcolare la media dei risultati
print(round(np.mean(result),2))

print("5% quantile =",np.percentile(result,5))
print("95% quantile =",np.percentile(result,95))

plt.hist(result,bins=100)
plt.axvline(np.percentile(result,5), color='r', linestyle='dashed', linewidth=2)
plt.axvline(np.percentile(result,95), color='r', linestyle='dashed', linewidth=2)
plt.show()