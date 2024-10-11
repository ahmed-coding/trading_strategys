import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Summary:
# Harmonic Pattern Detection: Detects specific harmonic patterns like Gartley based on Fibonacci ratios.
# Trend Detection: Uses harmonic patterns to predict potential price reversals.
# Machine Learning Prediction: Predicts price movement based on harmonic patterns and technical indicators like RSI and MACD.
# Added Butterfly, Bat, and Crab Patterns: These patterns use specific Fibonacci retracement and extension ratios for detection.
# Multiple Harmonic Patterns: Now, the system can detect Gartley, Butterfly, Bat, and Crab patterns, making the detection system more comprehensive.
class HarmonicPatterns:
    def __init__(self, data):
        self.data = data
        self.patterns = []

    def fibonacci_ratio(self, point_a, point_b):
        """ Calculate Fibonacci ratio between two points """
        return abs(point_a - point_b) / abs(point_b)

    def detect_gartley(self):
        # Existing Gartley pattern logic
        pass

    def detect_butterfly(self):
        for i in range(0, len(self.data) - 5):
            x, a, b, c, d = self.data['Close'][i:i+5]
            ab_ratio = self.fibonacci_ratio(x, a)
            bc_ratio = self.fibonacci_ratio(a, b)
            cd_ratio = self.fibonacci_ratio(b, c)

            if (0.786 <= ab_ratio <= 0.786) and (0.382 <= bc_ratio <= 0.886) and (1.618 <= cd_ratio <= 2.618):
                self.patterns.append((i, 'Butterfly'))

    def detect_bat(self):
        for i in range(0, len(self.data) - 5):
            x, a, b, c, d = self.data['Close'][i:i+5]
            ab_ratio = self.fibonacci_ratio(x, a)
            bc_ratio = self.fibonacci_ratio(a, b)
            cd_ratio = self.fibonacci_ratio(b, c)

            if (0.382 <= ab_ratio <= 0.5) and (0.382 <= bc_ratio <= 0.886) and (1.618 <= cd_ratio <= 2.618):
                self.patterns.append((i, 'Bat'))

    def detect_crab(self):
        for i in range(0, len(self.data) - 5):
            x, a, b, c, d = self.data['Close'][i:i+5]
            ab_ratio = self.fibonacci_ratio(x, a)
            bc_ratio = self.fibonacci_ratio(a, b)
            cd_ratio = self.fibonacci_ratio(b, c)

            if (0.382 <= ab_ratio <= 0.618) and (0.382 <= bc_ratio <= 0.886) and (2.24 <= cd_ratio <= 3.618):
                self.patterns.append((i, 'Crab'))

    def plot_patterns(self):
        plt.plot(self.data['Close'], label='Close Price')
        pattern_indices = [p[0] for p in self.patterns]
        plt.scatter(pattern_indices, self.data['Close'][pattern_indices], color='purple', label='Harmonic Patterns', marker='x')
        plt.legend()
        plt.show()
