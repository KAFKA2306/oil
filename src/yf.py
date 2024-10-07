import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YahooFinanceDataFetcher:
    def __init__(self, tickers_and_names, start_date, end_date):
        self.tickers_and_names = tickers_and_names
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        try:
            logging.info(f"Fetching data for {len(self.tickers_and_names)} tickers")
            data = yf.download(list(self.tickers_and_names.keys()) + ['JPY=X'], start=self.start_date, end=self.end_date)
            return self._process_data(data)
        except Exception as e:
            logging.error(f"An error occurred while fetching data: {e}")
            return None

    def _process_data(self, data):
        close_data = data['Close']
        close_data = close_data.ffill()  # Using ffill() instead of fillna(method='ffill')

        # Convert JPY prices to USD
        usd_jpy_rate = close_data['JPY=X']
        jpy_tickers = [ticker for ticker in self.tickers_and_names.keys() if '.T' in ticker]
        for ticker in jpy_tickers:
            close_data[ticker] = close_data[ticker] / usd_jpy_rate

        # Rename columns
        close_data.columns = [self.tickers_and_names.get(ticker, ticker) for ticker in close_data.columns]


        return close_data

    def save_data(self, data, filename):
        if data is not None:
            data.to_csv(filename)
            logging.info(f"Data has been successfully saved to {filename}")
        else:
            logging.error("Failed to save the data")

def main():
    tickers_and_names = {
        'CL=F': 'Crude Oil Futures',
        'BZ=F': 'Brent Crude Oil Futures',
        '5019.T': 'Idemitsu Kosan',
        '5020.T': 'ENEOS Holdings',
        '5021.T': 'Cosmo Energy Holdings',
        '5017.T': 'Fuji Oil Company',
        '8031.T': 'Mitsui & Co',
        '8002.T': 'Marubeni Corporation',
        '8058.T': 'Mitsubishi Corporation',
        '8053.T': 'Sumitomo Corporation',
        '9531.T': 'Tokyo Gas',
        '9532.T': 'Osaka Gas',
        '1605.T': 'INPEX Corporation',
        '9533.T': 'Toho Gas',
        '1662.T': 'Japan Petroleum Exploration',
        'XOM': 'ExxonMobil',
        'CVX': 'Chevron',
        'COP': 'ConocoPhillips',
        'HAL': 'Halliburton',
        'SLB': 'Schlumberger',
        '9101.T': 'Nippon Yusen',
        '9104.T': 'Mitsui O.S.K. Lines',
        '9107.T': 'Kawasaki Kisen Kaisha',
        '4005.T': 'Sumitomo Chemical',
        '4188.T': 'Mitsubishi Chemical Group',
        '4004.T': 'Showa Denko',
        '9501.T': 'Tokyo Electric Power',
        '9503.T': 'Kansai Electric Power',
        '9502.T': 'Chubu Electric Power',
        '9202.T': 'ANA Holdings',
        '9201.T': 'Japan Airlines',
        'JPY=X': 'USD/JPY Exchange Rate',
        'XOP': 'SPDR S&P Oil & Gas Exploration & Production ETF',
        'XLE': 'Energy Select Sector SPDR Fund',
        'OIH': 'VanEck Oil Services ETF',
        'USO': 'United States Oil Fund',
        'BP': 'BP plc',
        'E': 'Eni SpA',
        'EOG': 'EOG Resources',
        'VLO': 'Valero Energy Corporation',
        'PSX': 'Phillips 66',
        'MPC': 'Marathon Petroleum Corporation'
    }
    
    end_date = datetime.now()
    start_date = '2000-01-01'
    
    fetcher = YahooFinanceDataFetcher(tickers_and_names, start_date, end_date)
    data = fetcher.fetch_data()
    fetcher.save_data(data, "data/oil.csv")

if __name__ == "__main__":
    main()