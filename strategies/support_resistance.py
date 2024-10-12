import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Summary:
# Static Support and Resistance: Detects support and resistance levels using local minima and maxima in historical price data.
# Dynamic Support and Resistance: Uses moving averages to identify dynamic support and resistance levels.
# Machine Learning: Predicts price movements based on support and resistance levels, moving averages, and other indicators


class SupportResistance:
    def __init__(self, data):
        self.data = data
        self.support_levels = []
        self.resistance_levels = []

    def detect_support_resistance(self, window=20):
        """
        Detect static support and resistance levels by finding local minima (support)
        and local maxima (resistance) within a rolling window.
        """
        for i in range(window, len(self.data) - window):
            is_support = np.min(self.data['Low'][i - window:i + window]) == self.data['Low'][i]
            is_resistance = np.max(self.data['High'][i - window:i + window]) == self.data['High'][i]
            
            if is_support:
                self.support_levels.append((i, self.data['Low'][i]))
            if is_resistance:
                self.resistance_levels.append((i, self.data['High'][i]))

    def plot_support_resistance(self):
        plt.plot(self.data['Close'], label='Close Price')
        support_indices, support_prices = zip(*self.support_levels)
        resistance_indices, resistance_prices = zip(*self.resistance_levels)
        plt.scatter(support_indices, support_prices, color='green', marker='o', label='Support')
        plt.scatter(resistance_indices, resistance_prices, color='red', marker='x', label='Resistance')
        plt.legend()
        plt.show()



class DynamicSupportResistance:
    def __init__(self, data):
        self.data = data
        self.data['50_MA'] = self.data['Close'].rolling(window=50).mean()
        self.data['200_MA'] = self.data['Close'].rolling(window=200).mean()

    def plot_moving_averages(self):
        plt.plot(self.data['Close'], label='Close Price')
        plt.plot(self.data['50_MA'], label='50-Day MA', linestyle='--')
        plt.plot(self.data['200_MA'], label='200-Day MA', linestyle='--')
        plt.legend()
        plt.show()
