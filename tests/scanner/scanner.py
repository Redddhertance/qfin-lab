import pandas as pd
import yfinance as yf
#potential futuree info: Annualized Return (AR) Sharpe Ratio Max Drawdown Alpha (vs. Benchmark) Information Ratio
#define data source and extractors for  scanner function
tickers = pd.read_csv('tests/scanner/scticker.csv')['ticker'].tolist()
def analyse(tickers):
    data = []
    for ticker in tickers:
        print(f"Extracting data for {ticker}")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")

            #fundamental data
            roe = info.get('returnOnEquity', 0)
            if roe is None: roe = 0
            debt_to_equity = info.get('debtToEquity', 1000)
            if debt_to_equity is None: debt_to_equity = 1000
            pe_ratio = info.get('trailingPE', 0)
            if pe_ratio is None: pe_ratio = 0

            #technical data
            sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            if sma200 is None or pd.isna(sma200): sma200 = 0
            current_price = hist['Close'].iloc[-1]
            if current_price is None: current_price = 0
            trend = 'bull' if current_price > sma200 else 'bear'

            #decisionmaking
            signal = 'watch'
            if roe > 0.15 and debt_to_equity < 100 and pe_ratio < 35 and trend == 'bull':
                signal = 'buy'
            
            data.append({
                'ticker': ticker,
                'ROE': roe,
                'DebtToEquity': debt_to_equity,
                'PE_Ratio': pe_ratio,
                'Trend': trend,
                'Signal': signal
            })
        except Exception as e:
            print(f"Error extracting data for {ticker}: {e}")
    return pd.DataFrame(data)

#application
df = analyse(tickers)
#output results
print('Scanner Results:')
print(df[df['Signal'] == 'buy'])
df.to_csv('tests/scanner/scanner_results.csv')