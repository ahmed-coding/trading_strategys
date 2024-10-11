import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Divergence Detection: Identifies bullish and bearish divergence between price and momentum indicators (RSI, MACD, or Stochastic Oscillator).
# Signal Analysis: Helps spot potential trend reversals based on divergence signals.
# Machine Learning Prediction: Uses divergence signals and other market data to predict future price movements.


class Divergence:
    def __init__(self, data):
        self.data = data
        self.data['RSI'] = talib.RSI(self.data['Close'])
        self.data['MACD'], self.data['MACD_Signal'], _ = talib.MACD(self.data['Close'])
        self.data['Stochastic_K'], self.data['Stochastic_D'] = talib.STOCH(self.data['High'], self.data['Low'], self.data['Close'])
        self.divergences = []

    def detect_divergence(self, indicator='RSI'):
        """
        Detect bullish and bearish divergence between price and indicator (RSI, MACD, or Stochastic).
        """
        price_lows = self.data['Close'].rolling(window=10).min()
        indicator_lows = self.data[indicator].rolling(window=10).min()

        price_highs = self.data['Close'].rolling(window=10).max()
        indicator_highs = self.data[indicator].rolling(window=10).max()

        for i in range(1, len(self.data)):
            # Bullish divergence: Price making lower lows, indicator making higher lows
            if price_lows[i] < price_lows[i-1] and indicator_lows[i] > indicator_lows[i-1]:
                self.divergences.append((i, 'Bullish Divergence'))

            # Bearish divergence: Price making higher highs, indicator making lower highs
            if price_highs[i] > price_highs[i-1] and indicator_highs[i] < indicator_highs[i-1]:
                self.divergences.append((i, 'Bearish Divergence'))

    def plot_divergence(self, indicator='RSI'):
        plt.plot(self.data['Close'], label='Close Price')
        plt.plot(self.data[indicator], label=indicator)
        divergence_indices = [i[0] for i in self.divergences]
        plt.scatter(divergence_indices, self.data['Close'][divergence_indices], color='red', label='Divergence', marker='x')
        plt.legend()
        plt.show()
