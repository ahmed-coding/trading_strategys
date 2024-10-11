import matplotlib.pyplot as plt
from strategies.strategy import Strategy

class BreakoutPatterns(Strategy):
    def __init__(self, data):
        self.data = data
        self.support_levels = []
        self.resistance_levels = []

    def calculate_support_resistance(self, window=20):
        self.support_levels = self.data['Low'].rolling(window=window).min()
        self.resistance_levels = self.data['High'].rolling(window=window).max()

    def plot_support_resistance(self):
        plt.plot(self.data['Close'], label='Close Price')
        plt.plot(self.support_levels, label='Support', linestyle='--')
        plt.plot(self.resistance_levels, label='Resistance', linestyle='--')
        plt.legend()
        plt.show()
        
    def execute(self, data):
        # Same Breakout logic
        pass