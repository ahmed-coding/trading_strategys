import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Summary:
# Supply and Demand Zones: Detects price zones where strong buying or selling occurred, potentially leading to future reversals.
# Order Blocks: Key areas where large order executions create significant price reactions.
# Machine Learning Prediction: Predicts price movements based on supply and demand imbalances.


class SupplyDemand:
    def __init__(self, data):
        self.data = data
        self.supply_zones = []
        self.demand_zones = []

    def detect_zones(self, window=20, tolerance=0.02):
        """
        Detect supply and demand zones by analyzing price reversals.
        - window: Number of bars to look back to find reversals.
        - tolerance: Percentage difference to classify zones.
        """
        for i in range(window, len(self.data) - window):
            # Detect demand zones: Price reverses upwards after a drop
            if min(self.data['Low'][i - window:i]) == self.data['Low'][i] and np.mean(self.data['Close'][i+1:i+window]) > self.data['Close'][i] * (1 + tolerance):
                self.demand_zones.append((i, self.data['Low'][i]))

            # Detect supply zones: Price reverses downwards after a rise
            if max(self.data['High'][i - window:i]) == self.data['High'][i] and np.mean(self.data['Close'][i+1:i+window]) < self.data['Close'][i] * (1 - tolerance):
                self.supply_zones.append((i, self.data['High'][i]))

    def plot_zones(self):
        plt.plot(self.data['Close'], label='Close Price')

        # Plot supply zones in red and demand zones in green
        for i, price in self.supply_zones:
            plt.axhline(price, color='red', linestyle='--', alpha=0.7, label='Supply Zone' if i == 0 else "")

        for i, price in self.demand_zones:
            plt.axhline(price, color='green', linestyle='--', alpha=0.7, label='Demand Zone' if i == 0 else "")

        plt.legend()
        plt.show()
