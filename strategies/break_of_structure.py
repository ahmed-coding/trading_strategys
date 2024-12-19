import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Summary:
# Break of Structure Detection: Identifies when price breaks above resistance or below support, signaling potential trend reversals.
# Signal Analysis: Confirms trend continuation or reversal using BOS signals.
# Machine Learning: Trains a model to predict price movements based on BOS and other market data.


class BreakOfStructure:
    def __init__(self, data):
        self.data = data
        self.bos = []

    def detect_bos(self):
        """
        Detect break of structure (BOS) when the price breaks above a resistance level (uptrend)
        or below a support level (downtrend).
        """
        for i in range(2, len(self.data) - 2):
            if self.data['Close'].iloc[i] > max(self.data['High'].iloc[i-2:i]):
                self.bos.append((i, 'Break of Structure (Uptrend)'))
            elif self.data['Close'].iloc[i] < min(self.data['Low'].iloc[i-2:i]):
                self.bos.append((i, 'Break of Structure (Downtrend)'))

    def plot_bos(self):
        plt.plot(self.data['Close'], label='Close Price')
        bos_indices = [b[0] for b in self.bos]
        plt.scatter(bos_indices, self.data['Close'].iloc[bos_indices], color='red', label='Break of Structure', marker='x')
        plt.legend()
        plt.show()
