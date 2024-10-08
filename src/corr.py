import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def calculate_weekly_moving_average(df):
    return df.rolling(window=7).mean()

def calculate_returns(df):
    return df.pct_change().dropna()

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)

def plot_correlation_heatmap(returns, output_dir):
    corr_matrix = returns.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of All Assets', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

def plot_sharpe_vs_oil_correlation(returns, output_dir):
    oil_returns = returns['Crude Oil Futures']
    correlations = returns.corrwith(oil_returns)
    sharpe_ratios = returns.apply(calculate_sharpe_ratio)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(correlations, sharpe_ratios, c=sharpe_ratios, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    for i, asset in enumerate(returns.columns):
        plt.annotate(asset, (correlations[i], sharpe_ratios[i]), fontsize=8, alpha=0.7)
    
    plt.xlabel('Correlation with Crude Oil Futures')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio vs Correlation with Crude Oil Futures')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sharpe_vs_oil_correlation.png'), dpi=300)
    plt.close()

def analyze_rolling_correlation(returns, assets, windows, output_dir):
    for asset in assets:
        if asset != 'Crude Oil Futures':
            plt.figure(figsize=(12, 6))
            for window in windows:
                corr = returns['Crude Oil Futures'].rolling(window=window).corr(returns[asset])
                plt.plot(corr.index, corr, label=f'Window {window}')
            
            plt.title(f'Rolling Correlation: {asset} vs Crude Oil Futures')
            plt.xlabel('Date')
            plt.ylabel('Correlation')
            plt.legend()
            plt.tight_layout()
            safe_filename = "".join([c for c in asset if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            plt.savefig(os.path.join(output_dir, f"rolling_correlation_{safe_filename.replace(' ', '_').lower()}.png"))
            plt.close()

def analyze_usd_jpy_impact(returns, assets, windows, output_dir):
    usd_jpy_returns = returns['USD/JPY Exchange Rate']
    
    for asset in assets:
        if asset != 'USD/JPY Exchange Rate':
            plt.figure(figsize=(12, 6))
            for window in windows:
                corr = returns[asset].rolling(window=window).corr(usd_jpy_returns)
                plt.plot(corr.index, corr, label=f'Window {window}')
            
            plt.title(f'Rolling Correlation: {asset} vs USD/JPY')
            plt.xlabel('Date')
            plt.ylabel('Correlation')
            plt.legend()
            plt.tight_layout()
            safe_filename = "".join([c for c in asset if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            plt.savefig(os.path.join(output_dir, f"usd_jpy_correlation_{safe_filename.replace(' ', '_').lower()}.png"))
            plt.close()

def main():
    input_file = 'data/oil.csv'
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = load_data(input_file)
    if df is None:
        return
    
    # 週次移動平均の計算
    df_weekly = calculate_weekly_moving_average(df)
    
    # リターンの計算
    returns = calculate_returns(df_weekly)
    
    assets = list(returns.columns)
    windows = [40,50,60]  # 1週間、2週間、1ヶ月、2ヶ月、3ヶ月
    
    # 相関ヒートマップの作成
    plot_correlation_heatmap(returns, output_dir)
    
    # シャープレシオ vs 原油相関の散布図作成
    plot_sharpe_vs_oil_correlation(returns, output_dir)
    
    # ローリング相関分析
    #analyze_rolling_correlation(returns, assets, windows, output_dir)
    
    # USD/JPY為替レートの影響分析
    analyze_usd_jpy_impact(returns, assets, windows, output_dir)
    
    logging.info("Analysis complete. Results saved in the 'output' directory.")

def plot_correlation_heatmap(returns, output_dir):
    corr_matrix = returns.corr()
    
    # Sort the correlation matrix based on correlation with Crude Oil Futures
    sorted_corr = corr_matrix.sort_values('Crude Oil Futures', ascending=False)
    sorted_corr = sorted_corr.reindex(columns=sorted_corr.index)
    
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(sorted_corr, dtype=bool))
    sns.heatmap(sorted_corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True)
    plt.title('Correlation Heatmap of All Assets (Sorted by Crude Oil Futures Correlation)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap_sorted.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create a separate bar plot for Crude Oil Futures correlations
    oil_correlations = sorted_corr['Crude Oil Futures'].drop('Crude Oil Futures')
    plt.figure(figsize=(12, len(oil_correlations) * 0.3))
    oil_correlations.plot(kind='barh')
    plt.title('Correlations with Crude Oil Futures', fontsize=16)
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crude_oil_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print top 5 positive and negative correlations
    print("Top 5 Positive Correlations with Crude Oil Futures:")
    print(oil_correlations.nlargest(5))
    print("\nTop 5 Negative Correlations with Crude Oil Futures:")
    print(oil_correlations.nsmallest(5))
    
if __name__ == "__main__":
    main()