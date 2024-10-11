import talib
import pandas as pd
import matplotlib.pyplot as plt


# Summary:
# Oscillator Calculation: Calculates momentum oscillators like RSI, Stochastic Oscillator, and CCI to determine overbought or oversold market conditions.
# Trend Identification: Helps spot potential reversals or trend continuations using oscillator thresholds.
# Machine Learning: Uses oscillators and other market data to predict future price movements based on overbought/oversold conditions.


class Oscillators:
    def __init__(self, data):
        self.data = data
        self.data['RSI'] = talib.RSI(self.data['Close'])
        self.data['Stochastic_K'], self.data['Stochastic_D'] = talib.STOCH(self.data['High'], self.data['Low'], self.data['Close'])
        self.data['CCI'] = talib.CCI(self.data['High'], self.data['Low'], self.data['Close'])

    def plot_oscillators(self):
        plt.figure(figsize=(14, 10))
        
        # Plot Close Price
        plt.subplot(4, 1, 1)
        plt.plot(self.data['Close'], label='Close Price')
        plt.title('Close Price')
        plt.legend()
        
        # Plot RSI
        plt.subplot(4, 1, 2)
        plt.plot(self.data['RSI'], label='RSI', color='blue')
        plt.axhline(70, color='red', linestyle='--', label='Overbought')
        plt.axhline(30, color='green', linestyle='--', label='Oversold')
        plt.title('RSI')
        plt.legend()

        # Plot Stochastic Oscillator
        plt.subplot(4, 1, 3)
        plt.plot(self.data['Stochastic_K'], label='%K', color='purple')
        plt.plot(self.data['Stochastic_D'], label='%D', color='orange')
        plt.axhline(80, color='red', linestyle='--', label='Overbought')
        plt.axhline(20, color='green', linestyle='--', label='Oversold')
        plt.title('Stochastic Oscillator')
        plt.legend()

        # Plot CCI
        plt.subplot(4, 1, 4)
        plt.plot(self.data['CCI'], label='CCI', color='brown')
        plt.axhline(100, color='red', linestyle='--', label='Overbought')
        plt.axhline(-100, color='green', linestyle='--', label='Oversold')
        plt.title('Commodity Channel Index (CCI)')
        plt.legend()

        plt.tight_layout()
        plt.show()
