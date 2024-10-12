import matplotlib.pyplot as plt

# class Backtester:
#     def __init__(self, strategy, data):
#         self.strategy = strategy
#         self.data = data

#     def run_backtest(self):
#         # Apply the strategy and gather signals
#         self.strategy.execute(self.data)
#         # Example: Compare predicted vs actual, calculate profit/loss
#         self.plot_results()

#     def plot_results(self):
#         # Example plotting results of backtest
#         plt.plot(self.data['Close'], label='Actual Price')
#         # You could add simulated trades here
#         plt.legend()
#         plt.show()


class Backtester:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data

    def run_backtest(self):
        # Apply the strategy (e.g., ElliottWave)
        self.strategy.detect_waves()
        self.strategy.label_waves()
        self.strategy.plot_labeled_waves()

        # Simulate buy/sell based on wave patterns
        # For example, buy at troughs, sell at peaks
        initial_balance = 1000  # Start with $1000
        balance = initial_balance
        position = 0  # No open position at the start

        for idx, wave_type, label in self.strategy.waves:
            if "trough" in wave_type and position == 0:  # Buy at trough
                position = balance / self.data['Close'][idx]
                balance = 0
                print(f"Buying at {self.data['Close'][idx]}")
            elif "peak" in wave_type and position > 0:  # Sell at peak
                balance = position * self.data['Close'][idx]
                position = 0
                print(f"Selling at {self.data['Close'][idx]}")
        
        profit = balance - initial_balance
        print(f"Backtesting complete. Final Balance: {balance}, Profit: {profit}")