# Importing Required Libraries
import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mpl
import plotly.offline
import pylab
from scipy import stats

# Configurations of Libraries
pd.set_option('display.max_columns', 500)
mpl.rcParams['font.family'] = 'serif'
mpl.style.use('seaborn')

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
AMZN['Cumulative Returns'] = ((AMZN['Daily Returns']) + 1).cumprod()
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
srmp = AMZN_first * (1 + AMZN_simple_returns.mean()) ** len(AMZN_simple_returns)
print('Prediction of Last Price using Mean of Simple Returns: ', srmp)

# Using Product of Simple Returns to calculate latest price
srpp = AMZN_first * np.prod([(1 + R) for R in AMZN_simple_returns])
print('Prediction of Last Price using Product of Simple Returns: ', srpp)

# Log Returns have an additive property which is more useful since the sum is normally distributed
AMZN_log_returns = AMZN["Daily Log Returns"].dropna()
lrmp = AMZN_first * np.exp(AMZN_log_returns.mean() * len(AMZN_log_returns))
print("Prediction using Mean of Log Returns: ", lrmp)
print("Actual Price", AMZN_last)

# Plotting Histogram of Log Returns and Simple Returns
AMZN_log_returns.plot(kind='hist')
AMZN_simple_returns.plot(kind='hist')

# Normality
sorted_log_returns = AMZN_log_returns.tolist()
sorted_log_returns.sort()
print(sorted_log_returns)

worst = sorted_log_returns[0]
best = sorted_log_returns[1]

# Standardize
std_worst = (worst - AMZN_log_returns.mean()) / AMZN_log_returns.std()
std_best = (best - AMZN_log_returns.mean()) / AMZN_log_returns.std()
std = AMZN_log_returns.std()
print('Std : %.2f, Worst : %.2f, Best : %.2f' % (std, worst, best))

# Probability of Worst and Best Performance
prob_worst = stats.norm(0, 1).pdf(std_worst)
prob_best = stats.norm(0, 1).pdf(std_best)
print("The Probability of the worst return: %.15f" % prob_worst)
print("The Probability of the best return: %.10f" % prob_best)

# We can visually check whether the returns are normally distributed
# Q-Q Plot
stats.probplot(AMZN_log_returns, dist='norm', plot=pylab)

# Box Plot
AMZN_log_returns.plot(kind='box')

# statistical test and null hypthesis
# Ks Test
ks_stat, p_value = stats.kstest(AMZN_log_returns, 'norm')
print(ks_stat, p_value)

if p_value > 0.05:
    print("Gaussian")
else:
    print("NOT Gaussian")

# Shapiro Wills Test
sw_stat, p_value = stats.shapiro(AMZN_log_returns)
print(sw_stat, p_value)

if p_value > 0.05:
    print("Gaussian")
else:
    print("NOT Gaussian")


#function to calculate n period return
def n_period_price_return(price, n):
