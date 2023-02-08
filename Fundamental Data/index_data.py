import pandas as pd
import datetime as dt
import yfinance as yf
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Setting Start and End dates
end = dt.datetime.now().date()

# Importing SP500 Pricing
SP500 = yf.download('^GSPC', end=end)
SP500.dropna(inplace=True)

# Importing Nasdaq Composite Index
NDX = yf.download('^IXIC', end=end)
NDX.dropna(inplace=True)

# Importing DJI
DJI = yf.download('^DJI', end=end)
DJI.dropna(inplace=True)

# Importing Treasury Yield 30 Years
TYX = yf.download('^TYX', end=end)
TYX.dropna(inplace=True)

# Importing Treasury Yield 10 Years
TNX = yf.download('^TNX', end=end)
TNX.dropna(inplace=True)

# Importing Volatility Index
VIX = yf.download('^VIX', end=end)
VIX.dropna(inplace=True)

# Importing Russell 1000
R1000 = yf.download('^RUI', end=end)
R1000.dropna(inplace=True)

# Importing Russell 1000
R2000 = yf.download('^RUT', end=end)
R2000.dropna(inplace=True)

# Importing FTSE 100
FTSE100 = yf.download('^FTSE', end=end)
FTSE100.dropna(inplace=True)

# Importing Nikkie 225
N225 = yf.download('^N225', end=end)
N225.dropna(inplace=True)

# Importing Hang Seng Index
HSI = yf.download('^HSI', end=end)
HSI.dropna(inplace=True)
