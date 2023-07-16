import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# Funzione per scaricare i dati storici di una lista di ticker
def get_prices(tickers, start):
    prices = yf.download(tickers, start=start)['Adj Close']
    return prices
# Url dove scaricare le informazioni dei ticker che compongono il Dow 30
dow = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
# Leggere la pagina del sito e raccogliere i dati dei ticker
data_table = pd.read_html(dow)[1]
# Creazione della lista di ticker
tickers = data_table['Symbol'].tolist()
# tickers = data_table[:][0]['Symbol'].tolist()
# Download dei dati storici dei prezzi giornalieri per ogni ticker
prices = get_prices(tickers, start='2020-01-01')
# Calcolo dei rendimenti percentuali
returns = (((prices.iloc[-1] / prices.iloc[0]) - 1) * 100).round(2)

per_change = ((np.asarray(returns)).reshape(6,5))

fig, ax = plt.subplots(figsize=(14,9))
plt.title('Dow 30 Heat Map',fontsize=18)
ax.title.set_position([0.5,1.05])
ax.set_xticks([])
sns.heatmap(per_change, annot=True, fmt="", cmap='RdYlGn', ax=ax)
plt.show()


# Creazione dell'array dei ticker dei simboli con la stessa dimensione della heatmap
symbol = ((np.asarray(returns.index)).reshape(6,5))
# Creazione dell'array dei rendimenti percentuali con la stessa dimensione della heatmap
per_change = ((np.asarray(returns)).reshape(6,5))

# Creazione di un array che unisce i ticker ai rispettivi rendimenti percentuali
labels = (np.asarray(["{0} \n {1:.3f}".format(symbol, per_change)
                      for symbol, per_change in zip(symbol.flatten(),
                                               per_change.flatten())])).reshape(6,5)

fig, ax = plt.subplots(figsize=(14,9))
plt.title('Dow 30 Heat Map',fontsize=18)
ax.title.set_position([0.5,1.05])
ax.set_xticks([])
sns.heatmap(per_change, annot=labels, fmt="", cmap='RdYlGn', ax=ax)
plt.show()
print("")