import matplotlib.pyplot as plt
from strategies.strategy import Strategy

class FibonacciRetracement(Strategy):
    def __init__(self, data):
        self.data = data

    def calculate_levels(self):
        high = self.data['High'].max()
        low = self.data['Low'].min()
        diff = high - low
        levels = [high - (diff * ratio) for ratio in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]]
        return levels

    def plot_levels(self):
        levels = self.calculate_levels()
        plt.plot(self.data['Close'], label='Close Price')
        for level in levels:
            plt.axhline(level, linestyle='--', label=f'Fibonacci Level {level}')
        plt.legend()
        plt.show()
    def execute(self, data):
        # Same Breakout logic
        pass