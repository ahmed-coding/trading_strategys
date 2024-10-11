from binance.client import Client
import pandas as pd

API_KEY = 'your_binance_api_key'
API_SECRET = 'your_binance_secret_key'

class BinanceData:
    def __init__(self, symbol, interval, start_str):
        self.client = Client(API_KEY, API_SECRET)
        self.symbol = symbol
        self.interval = interval
        self.start_str = start_str
        self.data = self.get_binance_data()

    def get_binance_data(self):
        try:
            klines = self.client.get_historical_klines(self.symbol, self.interval, self.start_str)
        except Exception as e:
            print(f"Error fetching data from Binance: {e}")
            return None
        
        df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                           'Close time', 'Quote asset volume', 'Number of trades', 
                                           'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        df.set_index('Date', inplace=True)
        df.fillna(method='ffill', inplace=True)
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
