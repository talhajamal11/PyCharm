# Importing Required Libraries
import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mpl
import plotly.offline as plo
from pylab import plt

# Configurations of Libraries
pd.set_option('display.max_columns', 500)
mpl.rcParams['font.family'] = 'serif'
plt.style.use('seaborn')

start = dt.date(2012, 1, 1)
end = dt.datetime.now().date()

AMZN = yf.download('AMZN', start=start, end=end)
AMZN.dropna(inplace=True)
print(AMZN.head())

'''
Calculating Daily Returns, Log Returns, Cumulative Return  and Annualized Volatility
# Calculate Returns in Percentages - but when calculating Cumulative Returns make sure to convert it back to
# absolute values and not use percentage value in calculation
'''
AMZN['Daily Returns'] = AMZN['Adj Close'].pct_change()
AMZN['Daily Log Returns'] = np.log(AMZN['Adj Close'] / AMZN['Adj Close'].shift(1))
AMZN['Cumulative Returns'] = ((AMZN['Daily Returns'] ) + 1).cumprod()
AMZN['Annualized Volatility'] = AMZN['Daily Log Returns'].rolling(252).std() * np.sqrt(252)
print(AMZN.tail())

# Importing Benchmark
# start_date = AMZN.index[0].strftime("%Y-%m-%d")
SP500 = yf.download('^GSPC', start=start, end=end)
SP500.dropna(inplace=True)

# Calculating Daily Returns, Log Returns, Annualized Volatility and Cumulative Return for the Benchmark
# Calculate Returns in Percentages - but when calculating Cumulative Returns make sure to convert it back to
# absolute values and not use percentage value in calculation
SP500["Daily Returns"] = SP500["Adj Close"].pct_change()
SP500["Daily Log Returns"] = np.log(SP500["Adj Close"] / SP500["Adj Close"].shift(1))
SP500["Cumulative Returns"] = ((SP500["Daily Returns"]) + 1).cumprod()
SP500["Annualized Volatility"] = SP500["Daily Log Returns"].rolling(252).std() * np.sqrt(252)
print(SP500.tail())

'''
Plot Daily Log Returns, Annualized Volatility and Cumulative Returns
# AMZN[['Daily Log Returns', 'Annualized Volatility']].plot(subplots=True, figsize=(20, 10))
# AMZN[['Cumulative Returns']].plot(subplots=True, figsize=(20, 6))

# Plot Daily Log Returns, Annualized Volatility and Cumulative Returns for Benchmark
# SP500[['Daily Log Returns', 'Annualized Volatility']].plot(subplots=True, figsize=(20, 10))
# SP500[['Cumulative Returns']].plot(subplots=True, figsize=(20, 6))
'''
# Calculating SMA for AMZN
AMZN["21D SMA"] = AMZN["Adj Close"].rolling(21).mean()
AMZN["200D SMA"] = AMZN["Adj Close"].rolling(200).mean()

'''
Plot Stock Price and SMA to observe pattern
# AMZN[["Adj Close", "21D SMA", "200D SMA"]].plot(figsize=(20, 6))
'''
# Calculating SMA for the SP500
SP500["21D SMA"] = SP500["Adj Close"].rolling(21).mean()
SP500["200D SMA"] = SP500["Adj Close"].rolling(200).mean()

# Plot SP500 Daily Price and SMA to observe pattern
# SP500[["Adj Close", "21D SMA", "200D SMA"]].plot(figsize=(20, 10))

'''
 Why prefer Log Returns over Simple Returns?
 Main Reason: Modelling Returns using the Normal Distribution
 -> Simple Returns: The Product of Normally Distributed Variables is NOT normally distributed
 -> Log Returns: The sum of normally distributed variables DOES follow a normal distribution. 
                 The Log Distribution also bounds our stock price at 0 - which is a nice property to have and is 
                 consistent with reality
'''

AMZN_first = AMZN["Adj Close"][0]
AMZN_last = AMZN["Adj Close"][-1]

print('First Price', AMZN_first, 'Last Price', AMZN_last)

# Using Mean of Simple Returns to calculate latest price
AMZN_simple_returns = AMZN["Daily Returns"].dropna()
srmp = AMZN_first * (1 + AMZN_simple_returns.mean())**len(AMZN_simple_returns)
print('Prediction of Last Price using Mean of Simple Returns: ', srmp)

# Using Product of Simple Returns to calculate latest price
srpp = AMZN_first * np.prod([(1+R) for R in AMZN_simple_returns])
print('Prediction of Last Price using Product of Simple Returns: ', srpp)
