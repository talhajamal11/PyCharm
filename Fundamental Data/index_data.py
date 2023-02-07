import pandas as pd
import datetime as dt
import yfinance as yf
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Setting Start and End dates
end = dt.datetime.now().date()

#importing SP500 Pricing
SP500 = yf.download('^GSPC', end=end)
SP500.dropna(inplace=True)

