import talib
import pandas as pd
import matplotlib.pyplot as plt


# Summary:
# RSI, MACD, Stochastic Calculation: Calculates popular momentum indicators to detect overbought or oversold conditions.
# Trend Identification: Uses momentum indicators to spot potential reversals or continuations in market trends.
# Machine Learning Prediction: Trains a model to predict future price movements based on momentum indicators and other technical data.


class MomentumIndicators:
    def __init__(self, data):
        self.data = data
        self.data['RSI'] = talib.RSI(self.data['Close'])
        self.data['MACD'], self.data['MACD_Signal'], self.data['MACD_Hist'] = talib.MACD(self.data['Close'])
        self.data['Stochastic_K'], self.data['Stochastic_D'] = talib.STOCH(self.data['High'], self.data['Low'], self.data['Close'])

    def plot_indicators(self):
        plt.figure(figsize=(14, 8))
        
        # Plot MACD
        plt.subplot(3, 1, 1)
        plt.plot(self.data['Close'], label='Close Price')
        plt.title('Close Price')
        plt.legend()
        
        # Plot RSI
        plt.subplot(3, 1, 2)
        plt.plot(self.data['RSI'], label='RSI', color='orange')
        plt.axhline(70, color='red', linestyle='--', label='Overbought')
        plt.axhline(30, color='green', linestyle='--', label='Oversold')
        plt.title('RSI')
        plt.legend()

        # Plot MACD
        plt.subplot(3, 1, 3)
        plt.plot(self.data['MACD'], label='MACD')
        plt.plot(self.data['MACD_Signal'], label='MACD Signal')
        plt.fill_between(self.data.index, self.data['MACD_Hist'], 0, alpha=0.5, label='MACD Histogram')
        plt.title('MACD')
        plt.legend()

        plt.tight_layout()
        plt.show()
