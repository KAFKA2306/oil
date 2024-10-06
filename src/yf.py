import yfinance as yf
import pandas as pd

# 取得したいティッカーのリスト
tickers = [
    'CL=F', 'BZ=F', '5019.T', '5020.T', '5021.T', '5017.T', '8031.T', 
    '8002.T', '8058.T', '8053.T', '9531.T', '9532.T', '1605.T', 
    '9533.T', '1662.T', 'XOM', 'CVX', 'COP', 'HAL', 'SLB',
    '9101.T', '9104.T', '9107.T', '4005.T', '4188.T', '4004.T',
    '9501.T', '9503.T', '9502.T', '9202.T', '9201.T',
    'JPY=X', 'XOP', 'XLE'  # OVX, CRB, 1601.Tは取得失敗
]

try:
    # データをダウンロード（過去1年のデータを取得）
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
    
    # データをCSVファイルとして保存
    data.to_csv("./data/oil_related_data.csv")

    # 先頭の数行を表示
    print(data.head())
except Exception as e:
    print(f"エラーが発生しました: {e}")
