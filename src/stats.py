import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# データのロードとリターンの計算
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date')

def calculate_returns(df):
    return df.pct_change().dropna()

# 累積リターンのプロット
def plot_cumulative_returns(returns, title, output_dir):
    cumulative_returns = (1 + returns).cumprod()
    plt.figure(figsize=(12, 6))
    for col in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[col], label=col)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'))
    plt.close()

# フォルダの作成と準備
def prepare_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# メイン関数
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    output_dir = os.path.join(script_dir, '..', 'output')
    prepare_output_directory(output_dir)

    # データのロード
    df = load_data(os.path.join(data_dir, 'oil_related_data.csv'))

    # リターンの計算
    returns = calculate_returns(df)

    # 累積リターンのプロット
    plot_cumulative_returns(returns, 'Cumulative Returns for Oil-Related Stocks', output_dir)

    print("Analysis complete. Results and plots saved in the 'output' directory.")

if __name__ == "__main__":
    main()
