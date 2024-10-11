import pandas as pd
import matplotlib.pyplot as plt


# Summary:
# Volume Indicator Calculation: Calculates popular volume-based indicators like OBV, CMF, and VWAP to assess volume trends.
# Volume Analysis: Identifies potential breakout or reversal signals by analyzing the behavior of volume indicators.
# Machine Learning: Uses volume indicators and price data to predict future price movements and market trends.

class VolumeIndicators:
    def __init__(self, data):
        self.data = data
        self.data['OBV'] = self.calculate_obv()
        self.data['CMF'] = self.calculate_cmf()
        self.data['VWAP'] = self.calculate_vwap()

    def calculate_obv(self):
        """
        Calculate On-Balance Volume (OBV).
        """
        obv = [0]
        for i in range(1, len(self.data)):
            if self.data['Close'][i] > self.data['Close'][i-1]:
                obv.append(obv[-1] + self.data['Volume'][i])
            elif self.data['Close'][i] < self.data['Close'][i-1]:
                obv.append(obv[-1] - self.data['Volume'][i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=self.data.index)

    def calculate_cmf(self, period=20):
        """
        Calculate Chaikin Money Flow (CMF).
        """
        mf_multiplier = ((self.data['Close'] - self.data['Low']) - (self.data['High'] - self.data['Close'])) / (self.data['High'] - self.data['Low'])
        mf_volume = mf_multiplier * self.data['Volume']
        cmf = mf_volume.rolling(window=period).sum() / self.data['Volume'].rolling(window=period).sum()
        return cmf

    def calculate_vwap(self):
        """
        Calculate Volume Weighted Average Price (VWAP).
        """
        cumulative_vol = self.data['Volume'].cumsum()
        cumulative_vol_price = (self.data['Close'] * self.data['Volume']).cumsum()
        vwap = cumulative_vol_price / cumulative_vol
        return vwap

    def plot_volume_indicators(self):
        plt.figure(figsize=(14, 10))
        
        # Plot Close Price and VWAP
        plt.subplot(3, 1, 1)
        plt.plot(self.data['Close'], label='Close Price')
        plt.plot(self.data['VWAP'], label='VWAP', linestyle='--')
        plt.title('Close Price and VWAP')
        plt.legend()

        # Plot OBV
        plt.subplot(3, 1, 2)
        plt.plot(self.data['OBV'], label='OBV', color='orange')
        plt.title('On-Balance Volume (OBV)')
        plt.legend()

        # Plot CMF
        plt.subplot(3, 1, 3)
        plt.plot(self.data['CMF'], label='CMF', color='green')
        plt.axhline(0, color='red', linestyle='--', label='Neutral Line')
        plt.title('Chaikin Money Flow (CMF)')
        plt.legend()

        plt.tight_layout()
        plt.show()
