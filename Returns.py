import numpy as np
import matplotlib as mpl
from pylab import plt
import yfinance as yf
import pandas as pd
pd.set_option('display.max_columns', 500)

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

AMZN = yf.download('AMZN')
AMZN.dropna(inplace=True)
print(AMZN.head())

# Calculating Daily Returns, Log Returns, Cumulative Return  and Annualized Volatility
AMZN['Daily Returns'] = AMZN['Adj Close'].pct_change() * 100
AMZN['Daily Log Returns'] = np.log(AMZN['Adj Close'] / AMZN['Adj Close'].shift(1)) * 100
AMZN['Cumulative Returns'] = ((AMZN['Daily Returns'] / 100) + 1).cumprod()
AMZN['Annualized Volatility'] = AMZN['Daily Log Returns'].rolling(252).std() * np.sqrt(252)
print(AMZN.tail())

# Importing Benchmark
start_date = AMZN.index[0].strftime("%Y-%m-%d")
SP500 = yf.download('^GSPC', start=start_date)
SP500.dropna(inplace=True)

# Calculating Daily Returns, Log Returns, Annualized Volatility and Cumulative Return for the Benchmark
SP500["Daily Returns"] = SP500["Adj Close"].pct_change() * 100
SP500["Daily Log Returns"] = np.log(SP500["Adj Close"]/SP500["Adj Close"].shift(1)) * 100
SP500["Cumulative Returns"] = ((SP500["Daily Returns"] / 100) + 1).cumprod()
SP500["Annualized Volatility"] = SP500["Daily Log Returns"].rolling(252).std() * np.sqrt(252)
print(SP500.tail())


# Plot Daily Log Returns, Annualized Volatility and Cumulative Returns
AMZN[['Daily Log Returns', 'Annualized Volatility']].plot(subplots=True, figsize=(20, 10))
AMZN[['Cumulative Returns']].plot(subplots=True, figsize=(20, 6))

