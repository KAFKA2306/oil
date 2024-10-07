import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

TRADING_DAYS_PER_YEAR = 252
WINDOW_SIZE = 4 * TRADING_DAYS_PER_YEAR  # 4 years

def load_data(file_path):
    try:
        return pd.read_csv(file_path, index_col='Date', parse_dates=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    df = df.dropna(axis=1, how='all')
    earliest_date = df.dropna().index.min()
    print(f"Earliest date for normalization: {earliest_date}")
    return df.div(df.loc[earliest_date])

def calculate_returns(df):
    return df.pct_change().dropna()

def calculate_annual_metrics(returns):
    annual_return = returns.mean() * TRADING_DAYS_PER_YEAR
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = annual_return / annual_volatility
    return pd.DataFrame({'Annual Return': annual_return, 
                         'Annual Volatility': annual_volatility, 
                         'Sharpe Ratio': sharpe_ratio})

def plot_heatmap(data, title, output_dir):
    plt.figure(figsize=(20, 16))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

def select_stocks(returns, n_clusters=4, n_top_correlated=4):
    stock_returns = returns.drop('Crude Oil Futures', axis=1, errors='ignore')
    
    oil_correlation = stock_returns.corrwith(returns['Crude Oil Futures']).sort_values(ascending=False)
    top_correlated = oil_correlation.head(n_top_correlated).index.tolist()

    corr_matrix = stock_returns.rank().corr(method='spearman')
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    distance_matrix = squareform(1 - corr_matrix)
    clusters = hierarchy.fcluster(hierarchy.linkage(distance_matrix, method='ward'), 
                                  n_clusters, criterion='maxclust')

    metrics = calculate_annual_metrics(stock_returns)
    representatives = [metrics.loc[stock_returns.columns[clusters == i], 'Sharpe Ratio'].idxmax() 
                       for i in range(1, n_clusters + 1)]

    all_selected = list(dict.fromkeys(representatives + top_correlated))
    all_selected = metrics.loc[all_selected].sort_values(by='Sharpe Ratio', ascending=False).index.tolist()
    
    return all_selected + ['Crude Oil Futures']

def remove_outliers(series, n_std=3):
    z_scores = np.abs(stats.zscore(series))
    return series[z_scores < n_std]

def calculate_rolling_metrics(returns, window, benchmark_col='Crude Oil Futures', n_std=3):
    rolling_return = returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    rolling_volatility = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rolling_sharpe = rolling_return / rolling_volatility
    
    benchmark_returns = remove_outliers(returns[benchmark_col], n_std)
    aligned_returns = returns.loc[benchmark_returns.index]
    
    rolling_betas = pd.DataFrame()
    for column in aligned_returns.columns:
        if column != benchmark_col:
            cov = aligned_returns[column].rolling(window).cov(benchmark_returns)
            benchmark_var = benchmark_returns.rolling(window).var()
            rolling_betas[column] = cov / benchmark_var
    
    return rolling_return, rolling_volatility, rolling_sharpe, rolling_betas

def plot_time_series(data, title, ylabel, output_dir, filename, yscale='linear', ylim=None, hlines=None):
    plt.figure(figsize=(16, 8))
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if yscale == 'log':
        plt.yscale('log')
    plt.ylim(ylim)
    if hlines:
        for y, color in hlines:
            plt.axhline(y=y, color=color, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_rolling_metrics(returns, window, output_dir):
    rolling_return, rolling_volatility, rolling_sharpe, rolling_betas = calculate_rolling_metrics(returns, window)

    period_name = f"{window//TRADING_DAYS_PER_YEAR}-Year"
    
    plot_time_series(rolling_return, f'{period_name} Rolling Annual Returns', 'Annual Return', 
                     output_dir, f'{period_name.lower()}_rolling_annual_returns.png', ylim=(-0.1, 0.6), hlines=[(0, 'grey')])

    plot_time_series(rolling_sharpe, f'{period_name} Rolling Sharpe Ratio', 'Sharpe Ratio', 
                     output_dir, f'{period_name.lower()}_rolling_sharpe_ratio.png', ylim=(-0.6, 1.6), hlines=[(0, 'grey')])

def plot_stock_selection(returns, selected_stocks, output_dir):
    stock_returns = returns.drop('Crude Oil Futures', axis=1, errors='ignore')
    oil_correlation = stock_returns.corrwith(returns['Crude Oil Futures'])
    metrics = calculate_annual_metrics(stock_returns)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(oil_correlation, metrics['Sharpe Ratio'], 
                          c=metrics['Annual Volatility'], cmap='viridis', alpha=0.7)

    for stock in stock_returns.columns:
        plt.annotate(stock, (oil_correlation[stock], metrics.loc[stock, 'Sharpe Ratio']),
                     fontsize=8, alpha=0.7, ha='center', va='center')

    plt.colorbar(scatter, label='Annual Volatility')
    plt.xlabel('Correlation with Crude Oil')
    plt.ylabel('Sharpe Ratio')
    plt.title('Stock Selection: Oil Correlation vs Sharpe Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stock_selection_oil_correlation_vs_sharpe_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_rolling_correlation(returns, benchmark_col='Crude Oil Futures', window=WINDOW_SIZE):
    benchmark_returns = returns[benchmark_col]
    stock_returns = returns.drop(benchmark_col, axis=1)
    
    rolling_correlations = pd.DataFrame()
    for column in stock_returns.columns:
        rolling_correlations[column] = stock_returns[column].rolling(window=window).corr(benchmark_returns)
    
    return rolling_correlations

def plot_rolling_correlation(rolling_correlations, output_dir, filename='rolling_correlation_with_crude_oil_futures.png'):
    plt.figure(figsize=(16, 10))
    for column in rolling_correlations.columns:
        plt.plot(rolling_correlations.index, rolling_correlations[column], label=column)
    
    plt.title('Rolling Correlation with Crude Oil Futures', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()




def calculate_rolling_correlation(returns, benchmark_col='Crude Oil Futures', window=WINDOW_SIZE):
    benchmark_returns = returns[benchmark_col]
    stock_returns = returns.drop(benchmark_col, axis=1)
    
    rolling_correlations = pd.DataFrame()
    for column in stock_returns.columns:
        rolling_corr = stock_returns[column].rolling(window=252).corr(benchmark_returns)
        rolling_correlations[column] = rolling_corr
    
    return rolling_correlations


def main():
    input_file = 'data/oil.csv'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(input_file)
    if df is None:
        print("Failed to load data. Exiting.")
        return

    df_preprocessed = preprocess_data(df)
    returns = calculate_returns(df_preprocessed)

    plot_heatmap(df_preprocessed, 'Correlation Heatmap of All Stocks', output_dir)

    selected_stocks = select_stocks(returns)
    print(f"Selected stocks: {selected_stocks}")

    df_selected = df_preprocessed[selected_stocks]
    returns_selected = returns[selected_stocks]

    plot_time_series(df_selected, 'Normalized Stock Prices', 'Price (Normalized)', 
                     output_dir, 'normalized_prices.png', yscale='log', hlines=[(1, 'grey')])

    plot_rolling_metrics(returns_selected, WINDOW_SIZE, output_dir)

    metrics = calculate_annual_metrics(returns_selected)
    metrics.to_csv(os.path.join(output_dir, 'annual_metrics.csv'))

    plot_stock_selection(returns, selected_stocks, output_dir)

    rolling_correlations = calculate_rolling_correlation(returns_selected)
    plot_rolling_correlation(rolling_correlations, output_dir, 'rolling_correlation_with_crude_oil_futures.png')

    print("Analysis complete. Results saved in the 'output' directory.")


if __name__ == "__main__":
    main()
    
    
    