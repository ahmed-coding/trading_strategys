import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Summary:
# Trend Line Detection: Automatically detect uptrend and downtrend lines using price highs and lows.
# Price Action Analysis: Analyzes breakouts and potential trend reversals based on trend lines.
# Machine Learning: Predicts future price movements using trend lines and technical indicators such as RSI, volume, and moving averages.

# Summary of Enhancements:
# Breakout Detection: Adds breakout detection logic to signal potential buy/sell opportunities when prices cross trend lines.
# Multiple Trend Lines: Detects multiple trend lines at different timeframes for more detailed trend analysis.
# Dynamic Trend Line Adjustment: Adjusts trend lines dynamically based on evolving price data to maintain relevancy.

class TrendLines:
    def __init__(self, data):
        self.data = data
        self.uptrend_lines = []
        self.downtrend_lines = []

    def detect_trend_lines(self, window=20):
        """
        Detect trend lines by finding consecutive higher lows (uptrend)
        or consecutive lower highs (downtrend) within a rolling window.
        """
        lows = self.data['Low']
        highs = self.data['High']
        for i in range(window, len(lows) - window):
            # Check for uptrend (higher lows)
            if lows[i] > lows[i - window]:
                slope, intercept, _, _, _ = stats.linregress(range(window), lows[i - window:i])
                self.uptrend_lines.append((i, slope, intercept))

            # Check for downtrend (lower highs)
            if highs[i] < highs[i - window]:
                slope, intercept, _, _, _ = stats.linregress(range(window), highs[i - window:i])
                self.downtrend_lines.append((i, slope, intercept))
    def detect_breakouts(self):
            """
            Detect price breakouts above/below the identified trend lines.
            A breakout occurs when the price crosses above a downtrend line or below an uptrend line.
            """
            for i, (slope, intercept) in enumerate(self.uptrend_lines):
                trend_line_value = slope * i + intercept
                if self.data['Close'][i] < trend_line_value:
                    self.breakouts.append((i, 'breakdown'))
            
            for i, (slope, intercept) in enumerate(self.downtrend_lines):
                trend_line_value = slope * i + intercept
                if self.data['Close'][i] > trend_line_value:
                    self.breakouts.append((i, 'breakout'))

    def plot_breakouts(self):
            self.plot_trend_lines()  # Call the method to plot the existing trend lines
            breakout_indices = [i[0] for i in self.breakouts]
            plt.scatter(breakout_indices, self.data['Close'][breakout_indices], color='orange', label='Breakouts', marker='x')
            plt.legend()
            plt.show()
        
    def plot_trend_lines(self):
        plt.plot(self.data['Close'], label='Close Price')

        for i, slope, intercept in self.uptrend_lines:
            plt.plot(range(i - len(self.data['Low']), i), slope * np.arange(i - len(self.data['Low']), i) + intercept, color='green', label='Uptrend Line')

        for i, slope, intercept in self.downtrend_lines:
            plt.plot(range(i - len(self.data['High']), i), slope * np.arange(i - len(self.data['High']), i) + intercept, color='red', label='Downtrend Line')

        plt.legend()
        plt.show()
        
    def detect_multiple_trend_lines(self, window_sizes=[20, 50]):
        """
        Detect multiple trend lines with varying window sizes.
        """
        for window in window_sizes:
            self.detect_trend_lines(window=window)
            

    def adjust_trend_lines(self):
        """
        Adjust trend lines dynamically as new price data arrives.
        Recalculate slope and intercept periodically to adjust the lines.
        """
        self.detect_trend_lines()  # Redetect trend lines based on the new data
        self.detect_breakouts()  # Detect new breakouts after recalculating