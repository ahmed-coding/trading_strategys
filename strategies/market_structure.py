import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Summary:
# Market Structure Detection: Identifies higher highs, higher lows (uptrend), and lower highs, lower lows (downtrend) to analyze market trends.
# Break of Structure (BOS): Detects when price breaks through key support/resistance levels, signaling potential trend reversals.
# Machine Learning: Uses market structure and break of structure signals to predict future price movements.


class MarketStructure:
    def __init__(self, data):
        self.data = data
        self.highs = []
        self.lows = []

    def detect_structure(self):
        """
        Identify market structure by detecting higher highs, higher lows (uptrend),
        and lower highs, lower lows (downtrend).
        """
        for i in range(2, len(self.data) - 2):
            # Detect higher highs (HH) and higher lows (HL) for uptrend
            if self.data['High'][i] > max(self.data['High'][i-2:i]) and self.data['Low'][i] > min(self.data['Low'][i-2:i]):
                self.highs.append((i, self.data['High'][i]))
                self.lows.append((i, self.data['Low'][i]))
            # Detect lower highs (LH) and lower lows (LL) for downtrend
            elif self.data['High'][i] < min(self.data['High'][i-2:i]) and self.data['Low'][i] < min(self.data['Low'][i-2:i]):
                self.highs.append((i, self.data['High'][i]))
                self.lows.append((i, self.data['Low'][i]))

    def plot_structure(self):
        plt.plot(self.data['Close'], label='Close Price')
        highs_indices, highs_values = zip(*self.highs)
        lows_indices, lows_values = zip(*self.lows)
        plt.scatter(highs_indices, highs_values, color='green', label='Highs')
        plt.scatter(lows_indices, lows_values, color='red', label='Lows')
        plt.legend()
        plt.show()
