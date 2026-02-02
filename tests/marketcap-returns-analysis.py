import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
import numpy as np

#lowcap stocks
lowcap_tickers = pd.read_csv('data/IWM_holdings.csv', delimiter=',').fillna(0.0)

#Clean Market Value column by removing commas and converting to float, alongside removing invalid sectors and tickers
lowcap_tickers['Market Value'] = lowcap_tickers['Market Value'].str.replace(',', '').astype(float)
lowcap_tickers = lowcap_tickers[lowcap_tickers['Sector'] != 'Cash and/or Derivatives']
lowcap_tickers = lowcap_tickers[~lowcap_tickers['Ticker'].isin(['AEIS', 'MOGA'])]

#debugging purposes to ensure data has been removed
#print(lowcap_cap[lowcap_cap['Ticker'] == 'XTSLA'])
#print(lowcap_cap[lowcap_cap['Ticker'] == 'AEIS'])

lowcap_cap = lowcap_tickers[['Ticker', 'Market Value']]
lowcap_sample = lowcap_cap[:100]  #taking a sample of 100 tickers for testing purposes

#print(lowcap_cap.head())

ticker_list_low = lowcap_sample['Ticker'].tolist()
returntickerlow = yf.download(ticker_list_low, period='1y', auto_adjust=True)['Close']
returns_low= {}
for ticker in returntickerlow.columns:
    series = returntickerlow[ticker].dropna()
    #safety net
    if len(series) < 2:
        continue
    start_price = series.iloc[0]
    end_price = series.iloc[-1]
    returns_low[ticker] = (end_price / start_price) - 1
returns_low = pd.Series(returns_low)
#print(returns_low.head())

lowcap_merge = pd.merge(lowcap_sample, returns_low.rename('Return'), left_on='Ticker', right_index=True)
#print(lowcap_merge.head())
#highcap stocks

highcap_tickers = pd.read_csv('data/s&p500_highcap.csv', delimiter=',').fillna(0.0)
highcap_tickers = highcap_tickers.rename(columns={'Market Cap': 'Market Value'})
highcap_tickers = highcap_tickers.rename(columns={'Symbol': 'Ticker'})
highcap_tickers = highcap_tickers[~highcap_tickers['Ticker'].isin(['BF.B', 'ANSS', 'BRK.B'])]
highcap_cap = highcap_tickers[['Ticker', 'Market Value']]
highcap_sample = highcap_cap[:100]  #taking a sample of 100 tickers for testing purposes
returntickerhigh = yf.download(highcap_sample['Ticker'].tolist(), period='1y', auto_adjust=True)['Close']
returns_high= {}
for ticker in returntickerhigh.columns:
    series = returntickerhigh[ticker].dropna()
    #safety net
    if len(series) < 2:
        continue
    start_price = series.iloc[0]
    end_price = series.iloc[-1]
    returns_high[ticker] = (end_price / start_price) - 1
returns_high = pd.Series(returns_high)


highcap_merge = pd.merge(highcap_sample, returns_high.rename('Return'), left_on='Ticker', right_index=True)
print(highcap_merge.head())

plt.scatter(lowcap_merge['Market Value'], lowcap_merge['Return'], alpha=0.5)
plt.xscale('log')
plt.yscale('symlog')
#m, b = np.polyfit(lowcap_merge['Market Value'], lowcap_merge['Return'], 1)
#plt.plot(lowcap_merge['Market Value'], m*lowcap_merge['Market Value'] + b, color='red')
#simple polyfit, distorted due to log scaled axis so must do proper regression
#x = np.log10(lowcap_merge['Market Value'])
#y = lowcap_merge['Return']
x = lowcap_merge['Market Value']
y = lowcap_merge['Return']
m, b = np.polyfit(x, y, 1)
order = np.argsort(x)
x = x.iloc[order]
y_pred = m*x + b
plt.plot(x, m*x + b, color='red')
#plt.plot(lowcap_merge['Market Value'], 10**(m*np.log10(lowcap_merge['Market Value']) + b), color='red')
plt.xlabel('Market Value')
plt.ylabel('Return (%) (1 Year)')
plt.title('Low Cap Stocks: Market Cap vs Return')
plt.show()

plt.scatter(highcap_merge['Market Value'], highcap_merge['Return'], alpha=0.5)
plt.xscale('log')
plt.yscale('symlog')
#x = np.log10(highcap_merge['Market Value'])
#y = highcap_merge['Return']
x = highcap_merge['Market Value']
y = highcap_merge['Return']
m, b = np.polyfit(x, y, 1)
order = np.argsort(x)
x = x.iloc[order]
y_pred = m*x + b
plt.plot(x, m*x + b, color='red')
#plt.plot(highcap_merge['Market Value'], 10**(m*np.log10(highcap_merge['Market Value']) + b), color='red')
plt.xlabel('Market Value')
plt.ylabel('Return (%) (1 Year)')
plt.title('High Cap Stocks: Market Cap vs Return')
plt.show()