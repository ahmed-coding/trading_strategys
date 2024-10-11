import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Summary:
# Change of Character (CHoCH) Detection: Identifies market transitions from uptrend to downtrend, or vice versa.
# Signal Analysis: Analyzes potential trend reversals or continuations based on CHoCH.
# Machine Learning: Uses CHoCH signals and other market data to predict price movements.

class ChangeOfCharacter:
    def __init__(self, data):
        self.data = data
        self.choch_signals = []

    def detect_choch(self):
        """
        Detect Change of Character (CHoCH) when market transitions between
        higher highs/lows (bullish) to lower highs/lows (bearish), or vice versa.
        """
        for i in range(2, len(self.data) - 2):
            # Transition from uptrend to downtrend
            if self.data['High'][i] > max(self.data['High'][i-2:i]) and self.data['Low'][i] < min(self.data['Low'][i-2:i]):
                self.choch_signals.append((i, 'Change to Downtrend'))
            # Transition from downtrend to uptrend
            elif self.data['High'][i] < min(self.data['High'][i-2:i]) and self.data['Low'][i] > max(self.data['Low'][i-2:i]):
                self.choch_signals.append((i, 'Change to Uptrend'))

    def plot_choch(self):
        plt.plot(self.data['Close'], label='Close Price')
        choch_indices = [s[0] for s in self.choch_signals]
        plt.scatter(choch_indices, self.data['Close'][choch_indices], color='blue', label='CHoCH', marker='x')
        plt.legend()
        plt.show()
