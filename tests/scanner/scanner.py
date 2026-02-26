import pandas as pd
import yfinance as yf
import numpy as np
#potential future info: Annualized Return (AR) Alpha (vs. Benchmark)
#ignored features: information ratio due to lack of usefulness, complex stress-tests due to unreliability and very very inconsistent and inapplicable market assumptions
#define data source and extractors for  scanner function
tickers = pd.read_csv('tests/scanner/scticker.csv')['ticker'].tolist()
def bulk_download(ticker_list):
    data = yf.download(ticker_list, period='1y', auto_adjust=True, threads= True, group_by='ticker', progress=True)
    survivors = []
    for ticker in ticker_list:
        if ticker not in data.columns.levels[0]:
            continue
                
        df = data[ticker]
        if df['Close'].isna().all() or len(df) < 200:
            continue
        prices = df['Close']
        current_price = prices.iloc[-1]
        sma200 = prices.rolling(window=200).mean().iloc[-1]
        if current_price > sma200:
            survivors.append(ticker)
    print(f"Filter successful. Reducied  {len(ticker_list)} -> {len(survivors)} candidates.")
    return survivors



def screener(tickers): #fundamental/quant screener function which prints to scanner_results.csv
    data = []
    for ticker in tickers:
        print(f"Extracting data for {ticker}")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            if hist.empty:
                print(f" No historical data found for {ticker}")
                continue
            #fundamental data
            roe = info.get('returnOnEquity', 0)
            if roe is None: roe = 0
            debt_to_equity = info.get('debtToEquity') 
            if debt_to_equity is None: debt_to_equity = 1000 #default to reduce weight if data is missing
            pe_ratio = info.get('trailingPE', 0)
            if pe_ratio is None: pe_ratio = 0

            #technical data
            sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            if sma200 is None or pd.isna(sma200): sma200 = 0
            current_price = hist['Close'].iloc[-1]
            if current_price is None: current_price = 0
            trend = 'bull' if current_price > sma200 else 'bear'
            Beta = info.get('beta', 1)
            if Beta is None: Beta = 1

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
                'Signal': signal,
                'Beta': Beta
            })
        except Exception as e:
            print(f"Failed {ticker} Reason: {e}")
    return pd.DataFrame(data)

def riskmetrics(tickers): #risk metrics function which prints to riskmetrics_results.csv
    results = []
    sector_data = yf.Ticker("XLC").history(period="1y") #using Communication Services sector ETF as benchmark for beta calculation, adjust different sectors
    for t in tickers:
        print(f'Extracting risk metrics for {t}')

        try:
            stock = yf.Ticker(t)
            stock_metadata = stock.info
            hist = stock.history(period="1y")
            if hist.empty:
                print(f" No historical data found for {t}")
                continue
            hist['Returns'] = hist['Close'].pct_change()
            volatility = hist['Returns'].std() * np.sqrt(252)
            total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
            risk_free = 0.04
            sharpe_ratio = (total_return - risk_free) / volatility if volatility != 0 else 0
            rolling_max = hist['Close'].cummax()
            daily_drawdown = hist['Close'] / rolling_max - 1
            max_dd = daily_drawdown.min()
            var_95 = np.percentile(hist['Returns'].dropna(), 5)
            cvar_95 = hist['Returns'][hist['Returns'] <= var_95].mean()
            beta = stock_metadata.get('beta', 1) #using Communication Services sector ETF as benchmark for beta calculation, adjust different sectors
            sector_comparison = pd.DataFrame({
                'Stock' : hist['Returns'],
                'Sector': sector_data['Close'].pct_change()
            }).dropna()
            sector_shock = -0.10 #assumed 10% sector correction
            matrix = np.cov(sector_comparison['Stock'], sector_comparison['Sector'])
            sectorbeta = matrix[0, 1] / matrix[1, 1]
            sector_risk = sectorbeta * sector_shock
            scenarios = {
                'market_correction10': -0.10,
                'market_drop20' : -0.20,
                'market_crash30' : -0.30
            } #ADJUST FOR SENSITIVITY ANALYSIS SCENARIOS (not currently used in report due to inaccurate and unreliable assumptions, var and cvar probbaly more effective for determining risk exposure in report)
            stress_results = {}
            for scenario, market_drop in scenarios.items():
                expected_drop = beta * market_drop
                stress_results[scenario] = f"{expected_drop:.2%}"
            results.append({
                'ticker': t,
                'volatility': f"{volatility:.2%}",
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': f"{max_dd:.2%}",
                'beta': beta,
                'Sensitivity Analysis:': stress_results,
                'VaR_95': f"{var_95:.2%}",
                'CVaR_95': f"{cvar_95:.2%}",
                'Sector Beta': f"{sectorbeta:.2f}",
                'Sector Risk Exposure': f"{sector_risk:.2%}"
                })
            for scenario, market_drop in scenarios.items():
                expected_drop = beta * market_drop
                stress_results[scenario] = f"{expected_drop:.2%}"
        except Exception as e:
            print(f"Failed: {t} with error: {e}")
    return pd.DataFrame(results)
#application
while True:
    mode = input('([S] Scanner / [R] Risk Metrics, [A] All): ').strip().upper()
    if mode in ['S', 'R', 'A']:
        break
    print('Invalid input. Please enter S, R, or A.')
print(f'Selected mode: {mode}')
if mode == 'R':
    filtered = bulk_download(tickers)
    df = riskmetrics(filtered)
    print('Risk Metrics Results:')
    print(df)
    df.to_csv('tests/scanner/riskmetrics_results.csv')
elif mode == 'S':
    filtered = bulk_download(tickers)
    df = screener(filtered)
    print('Scanner Results:')
    print(df[df['Signal'] == 'buy'])
    df.to_csv('tests/scanner/scanner_results.csv')
elif mode == 'A':
    filtered = bulk_download(tickers)
    df1 = screener(filtered)
    print('Scanner Results:')
    print(df1[df1['Signal'] == 'buy'])
    df1.to_csv('tests/scanner/scanner_results.csv')
    df2 = riskmetrics(filtered)
    print('Risk Metrics Results:')
    print(df2)
    df2.to_csv('tests/scanner/riskmetrics_results.csv')