import pandas as pd
import matplotlib.pyplot as plt

class HeikinAshi:
    def __init__(self, data):
        self.data = data
        self.ha_data = pd.DataFrame(index=data.index)

    def calculate_heikin_ashi(self):
        self.ha_data['HA_Close'] = (self.data['Open'] + self.data['High'] + self.data['Low'] + self.data['Close']) / 4
        self.ha_data['HA_Open'] = ((self.data['Open'].shift(1) + self.data['Close'].shift(1)) / 2).fillna(self.data['Open'])
        self.ha_data['HA_High'] = self.data[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        self.ha_data['HA_Low'] = self.data[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

    def plot_heikin_ashi(self):
        plt.plot(self.data['Close'], label='Close Price')
        plt.plot(self.ha_data['HA_Close'], label='Heikin Ashi Close', linestyle='--')
        plt.fill_between(self.ha_data.index, self.ha_data['HA_Low'], self.ha_data['HA_High'], color='gray', alpha=0.3)
        plt.legend()
        plt.show()
