import pandas as pd
import matplotlib.pyplot as plt

class FairValueGap:
    def __init__(self, data):
        self.data = data
        self.fvg_list = []

    def identify_fvg(self):
        """
        Identifies fair value gaps by checking if the high of one candle is lower
        than the low of the next candle.
        """
        for i in range(1, len(self.data)):
            if self.data['High'][i-1] < self.data['Low'][i]:
                gap = {
                    'index': i,
                    'high': self.data['High'][i-1],
                    'low': self.data['Low'][i]
                }
                self.fvg_list.append(gap)

    def plot_fvg(self):
        """
        Plot the fair value gaps (FVG) on the price chart.
        """
        plt.plot(self.data['Close'], label='Close Price')
        for gap in self.fvg_list:
            plt.axhline(gap['high'], color='red', linestyle='--', label='FVG High')
            plt.axhline(gap['low'], color='blue', linestyle='--', label='FVG Low')
        plt.legend()
        plt.show()
