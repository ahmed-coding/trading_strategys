import matplotlib.pyplot as plt
from strategies.strategy import Strategy
# import talib
from ta.momentum import rsi


class BreakoutPatterns:
    def __init__(self, data, ml_model):
        self.data = data
        self.support_levels = []
        self.resistance_levels = []
        self.signals = []
        self.ml_model = ml_model  # Add ML model for trade validation

        # Calculate additional features needed for ML
        self.data['Price_Change'] = self.data['Close'].diff()  # Add Price_Change to the strategy
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        # self.data['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)
        self.data['RSI'] = rsi(self.data['Close'], window=14)
        self.data['Volume_Change'] = self.data['Volume'].diff()

    def calculate_support_resistance(self, window=20):
        """
        Calculate the rolling support and resistance levels using a specified window.
        """
        self.support_levels = self.data['Low'].rolling(window=window).min().ffill()
        self.resistance_levels = self.data['High'].rolling(window=window).max().ffill()

    def plot_support_resistance(self):
        """
        Plot the support and resistance levels along with the closing price.
        """
        plt.plot(self.data['Close'], label='Close Price')
        plt.plot(self.support_levels, label='Support', linestyle='--', color='green')
        plt.plot(self.resistance_levels, label='Resistance', linestyle='--', color='red')
        plt.legend()
        plt.show()

    def execute(self):
        """
        Execute breakout strategy with ML model confirmation.
        """
        self.calculate_support_resistance()

        for i in range(1, len(self.data)):
            # Prepare features for ML model
            features = [
                self.data['Price_Change'].iloc[i],
                self.data['SMA_10'].iloc[i],
                self.data['SMA_50'].iloc[i],
                self.data['RSI'].iloc[i],
                self.data['Volume_Change'].iloc[i]
            ]

            # Detect breakout and validate with ML model
            if self.data['Close'].iloc[i] > self.resistance_levels.iloc[i]:
                ml_prediction = self.ml_model.predict(features)
                if ml_prediction == 1:  # ML model confirms 'Buy' signal
                    print(f"ML-Confirmed Breakout: Buy at {self.data.index[i]} Price: {self.data['Close'].iloc[i]}")
                    self.signals.append((self.data.index[i], self.data['Close'].iloc[i], 'Buy'))

            elif self.data['Close'].iloc[i] < self.support_levels.iloc[i]:
                ml_prediction = self.ml_model.predict(features)
                if ml_prediction == 0:  # ML model confirms 'Sell' signal
                    print(f"ML-Confirmed Breakdown: Sell at {self.data.index[i]} Price: {self.data['Close'].iloc[i]}")
                    self.signals.append((self.data.index[i], self.data['Close'].iloc[i], 'Sell'))
