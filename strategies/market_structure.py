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


    def detect_break_of_structure(self):
        """
        Detect break of structure (BOS) when price breaks above a resistance level (higher high)
        or below a support level (lower low).
        """
        self.bos = []
        for i in range(1, len(self.highs)):
            if self.data['Close'][self.highs[i][0]] < self.data['Close'][self.highs[i-1][0]]:
                self.bos.append((self.highs[i][0], 'Break of Structure (Uptrend)'))
            elif self.data['Close'][self.lows[i][0]] > self.data['Close'][self.lows[i-1][0]]:
                self.bos.append((self.lows[i][0], 'Break of Structure (Downtrend)'))

    def plot_bos(self):
        self.plot_structure()
        bos_indices = [bos[0] for bos in self.bos]
        plt.scatter(bos_indices, self.data['Close'][bos_indices], color='orange', label='Break of Structure', marker='x')
        plt.legend()
        plt.show()