import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TRADING_DAYS_PER_YEAR = 252
WINDOW_SIZE = 2 * TRADING_DAYS_PER_YEAR
RISK_FREE_RATE = 0.001
CACHE_DIR = 'data_cache'
MAX_WORKERS = 3
DELAY_BETWEEN_REQUESTS = 2  # seconds

# Japan Top 100 Companies
JAPAN_TOP_100 = {
    '7203.T': 'Toyota Motor', '8306.T': 'MUFG', '6758.T': 'Sony Group',
    '6861.T': 'Keyence', '6098.T': 'Recruit Holdings', '9432.T': 'NTT',
    '9983.T': 'Fast Retailing', '8035.T': 'Tokyo Electron', '4063.T': 'Shin-Etsu Chemical',
    # ... (残りの企業は省略しますが、実際のスクリプトには全て含めてください)
}

# Benchmark options
BENCHMARK_OPTIONS = ['1321.T', '^N225', '1306.T']  # Nikkei 225 ETF, Nikkei 225 Index, TOPIX ETF

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def get_cached_data(ticker, start_date, end_date):
    cache_file = os.path.join(CACHE_DIR, f"{ticker.replace('^', '')}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index = df.index.tz_localize(None)
        if df.index[0].date() <= start_date.date() and df.index[-1].date() >= end_date.date():
            return df
    return None

def save_to_cache(ticker, df):
    cache_file = os.path.join(CACHE_DIR, f"{ticker.replace('^', '')}.csv")
    df.to_csv(cache_file)

def fetch_data(ticker, start_date, end_date):
    try:
        cached_data = get_cached_data(ticker, start_date, end_date)
        if cached_data is not None:
            return ticker, cached_data

        time.sleep(DELAY_BETWEEN_REQUESTS)
        data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        if not data.empty:
            save_to_cache(ticker, data)
            return ticker, data
        else:
            logging.warning(f"No data found for {ticker}")
            return ticker, None
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return ticker, None

def load_data(tickers, start_date, end_date):
    ensure_cache_dir()
    all_data = {}
    failed_tickers = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(fetch_data, ticker, start_date, end_date): ticker for ticker in list(tickers.keys()) + BENCHMARK_OPTIONS}
        for future in as_completed(future_to_ticker):
            ticker, data = future.result()
            if data is not None:
                all_data[ticker] = data
            else:
                failed_tickers.append(ticker)

    df = pd.DataFrame(all_data)
    df.columns = [tickers.get(col, col) for col in df.columns]
    return df.ffill().dropna(axis=1), failed_tickers

def select_benchmark(returns):
    for candidate in BENCHMARK_OPTIONS:
        if candidate in returns.columns:
            logging.info(f"Using {candidate} as benchmark")
            return candidate
    raise ValueError("No suitable benchmark found. Analysis cannot proceed.")

def calculate_returns(df):
    return df.pct_change().dropna()

def calculate_rolling_metrics(returns, benchmark_returns, window):
    excess_returns = returns.sub(benchmark_returns, axis=0)
    
    rolling_return = returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    rolling_volatility = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rolling_sharpe = (rolling_return - RISK_FREE_RATE) / rolling_volatility
    
    rolling_beta = returns.rolling(window).apply(lambda x: np.cov(x, benchmark_returns.loc[x.index])[0, 1] / np.var(benchmark_returns.loc[x.index]))
    rolling_alpha = rolling_return - RISK_FREE_RATE - (rolling_beta * (benchmark_returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR - RISK_FREE_RATE))
    
    rolling_sortino = (rolling_return - RISK_FREE_RATE) / (returns[returns < 0].rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    
    return rolling_sharpe, rolling_beta, rolling_alpha, rolling_sortino

def plot_rolling_metrics(metrics, metric_name, top_n, output_dir):
    plt.figure(figsize=(16, 10))
    for column in metrics.iloc[-1].nlargest(top_n).index:
        plt.plot(metrics.index, metrics[column], label=column)
    
    plt.title(f'Rolling {metric_name} - Top {top_n} Stocks', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rolling_{metric_name.lower().replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_risk_adjusted_performance(returns, benchmark_returns):
    excess_returns = returns.sub(benchmark_returns, axis=0)
    
    sharpe_ratio = (excess_returns.mean() * TRADING_DAYS_PER_YEAR) / (excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sortino_ratio = (excess_returns.mean() * TRADING_DAYS_PER_YEAR) / (excess_returns[excess_returns < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    
    beta = returns.apply(lambda x: np.cov(x, benchmark_returns)[0, 1] / np.var(benchmark_returns) if not x.empty else np.nan)
    alpha = returns.mean() * TRADING_DAYS_PER_YEAR - RISK_FREE_RATE - beta * (benchmark_returns.mean() * TRADING_DAYS_PER_YEAR - RISK_FREE_RATE)
    
    return pd.DataFrame({
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Beta': beta,
        'Alpha': alpha
    })

def plot_correlation_heatmap(returns, output_dir):
    plt.figure(figsize=(20, 16))
    sns.heatmap(returns.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Stock Returns', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    end_date = datetime.now().replace(tzinfo=None)
    start_date = end_date - timedelta(days=5*365)
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    df, failed_tickers = load_data(JAPAN_TOP_100, start_date, end_date)
    returns = calculate_returns(df)
    
    # Log failed tickers
    if failed_tickers:
        logging.warning(f"Failed to fetch data for the following tickers: {', '.join(failed_tickers)}")
    
    # Select benchmark
    benchmark_name = select_benchmark(returns)
    benchmark_returns = returns[benchmark_name]
    stock_returns = returns.drop(benchmark_name, axis=1)
    
    # Calculate rolling metrics
    rolling_sharpe, rolling_beta, rolling_alpha, rolling_sortino = calculate_rolling_metrics(stock_returns, benchmark_returns, WINDOW_SIZE)
    
    # Plot rolling metrics
    plot_rolling_metrics(rolling_sharpe, 'Sharpe Ratio', 10, output_dir)
    plot_rolling_metrics(rolling_beta, 'Beta', 10, output_dir)
    plot_rolling_metrics(rolling_alpha, 'Alpha', 10, output_dir)
    plot_rolling_metrics(rolling_sortino, 'Sortino Ratio', 10, output_dir)
    
    # Calculate risk-adjusted performance
    risk_adjusted_performance = calculate_risk_adjusted_performance(stock_returns, benchmark_returns)
    
    # Plot correlation heatmap
    plot_correlation_heatmap(stock_returns, output_dir)
    
    # Print final results
    print("\nTop 10 Stocks by Sharpe Ratio:")
    print(risk_adjusted_performance['Sharpe Ratio'].nlargest(10))
    
    print("\nTop 10 Stocks by Sortino Ratio:")
    print(risk_adjusted_performance['Sortino Ratio'].nlargest(10))
    
    print("\nTop 10 Stocks by Alpha:")
    print(risk_adjusted_performance['Alpha'].nlargest(10))
    
    print("\nTop 10 Stocks by Beta (lowest):")
    print(risk_adjusted_performance['Beta'].nsmallest(10))
    
    # Save risk-adjusted performance to CSV
    risk_adjusted_performance.to_csv(os.path.join(output_dir, 'risk_adjusted_performance.csv'))
    
    logging.info("Analysis complete. Results saved in the 'output' directory.")

if __name__ == "__main__":
    main()