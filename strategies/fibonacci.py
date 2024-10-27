import matplotlib.pyplot as plt
from strategies.strategy import Strategy
import talib
import pandas as pd
import numpy as np

# class FibonacciStrategy:
#     def __init__(self, data, ml_model):
#         self.data = data
#         self.levels = []
#         self.signals = []
#         self.ml_model = ml_model  # Add ML model for trade validation

#     def calculate_levels(self):
#         """
#         Calculate Fibonacci retracement levels based on the highest high and lowest low in the dataset.
#         """
#         high = self.data['High'].max()
#         low = self.data['Low'].min()
#         diff = high - low
#         self.levels = [high - (diff * ratio) for ratio in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]]

#     def plot_levels(self):
#         """
#         Plot the Fibonacci retracement levels along with the closing price.
#         """
#         plt.plot(self.data['Close'], label='Close Price')
#         for level in self.levels:
#             plt.axhline(level, linestyle='--', label=f'Fibonacci Level {level:.2f}')
#         plt.legend()
#         plt.show()

#     def execute(self):
#         """
#         Execute trade logic based on Fibonacci retracement levels.
#         Entry: When price retraces to a Fibonacci level and shows reversal.
#         Exit: When price hits another Fibonacci level or breaks below.
#         """
#         self.calculate_levels()

#         for i in range(1, len(self.data)):
#             for level in self.levels:
#                 # Detect if price reaches a Fibonacci level within a 1% threshold
#                 if level * 0.99 <= self.data['Close'].iloc[i] <= level * 1.01:
#                     print(f"Detected price near Fibonacci level {level:.2f} on {self.data.index[i]} at price {self.data['Close'].iloc[i]}")

#                     # Prepare features for ML model
#                     features = [
#                         self.data['Price_Change'].iloc[i],
#                         self.data['SMA_10'].iloc[i],
#                         self.data['SMA_50'].iloc[i],
#                         self.data['RSI'].iloc[i],
#                         self.data['Volume_Change'].iloc[i]
#                     ]

#                     # Check ML model prediction
#                     ml_prediction = self.ml_model.predict(features)
#                     if ml_prediction == 1:  # ML model confirms 'Buy' signal
#                         print(f"ML-Confirmed Buy Signal at Fibonacci level {level:.2f} on {self.data.index[i]} at price {self.data['Close'].iloc[i]}")
#                         self.signals.append((self.data.index[i], self.data['Close'].iloc[i], 'Buy'))
#                     elif ml_prediction == 0:  # ML model confirms 'Sell' signal
#                         print(f"ML-Confirmed Sell Signal at Fibonacci level {level:.2f} on {self.data.index[i]} at price {self.data['Close'].iloc[i]}")
#                         self.signals.append((self.data.index[i], self.data['Close'].iloc[i], 'Sell'))



class FibonacciStrategy:
    def __init__(self, data, ml_model, scaler):
        self.data = data
        self.levels = []
        self.signals = []
        self.ml_model = ml_model  # Add ML model for trade validation
        self.scaler = scaler  # Add the scaler here

        # Calculate additional features needed for ML and strategy
        self.data['Price_Change'] = self.data['Close'].diff()  # Calculate Price_Change
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()  # Calculate SMA 10
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()  # Calculate SMA 50
        self.data['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)  # Calculate RSI
        self.data['Volume_Change'] = self.data['Volume'].diff()  # Calculate Volume Change

    def calculate_levels(self):
        """
        Calculate Fibonacci retracement levels based on the highest high and lowest low in the dataset.
        """
        high = self.data['High'].max()
        low = self.data['Low'].min()
        diff = high - low
        self.levels = [high - (diff * ratio) for ratio in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]]

    def plot_levels(self):
        """
        Plot the Fibonacci retracement levels along with the closing price.
        """
        plt.plot(self.data['Close'], label='Close Price')
        for level in self.levels:
            plt.axhline(level, linestyle='--', label=f'Fibonacci Level {level:.2f}')
        plt.legend()
        plt.show()



    def execute(self):
        self.calculate_levels()

        # Ensure the correct features are used for fitting the scaler
        features_for_scaling = ['Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI', 'Price_Change', 'Volume_Change']

        if not hasattr(self.scaler, 'mean_'):
            # Fit the scaler if it's not already fitted
            self.scaler.fit(self.data[features_for_scaling])

        for i in range(1, len(self.data)):
            for level in self.levels:
                if level * 0.99 <= self.data['Close'].iloc[i] <= level * 1.01:
                    print(f"Detected price near Fibonacci level {level:.2f} on {self.data.index[i]} at price {self.data['Close'].iloc[i]}")

                    # Select the correct features for the prediction
                    features = [
                        self.data['Close'].iloc[i],         # 'Close'
                        self.data['Volume'].iloc[i],        # 'Volume'
                        self.data['SMA_10'].iloc[i],        # 'SMA_10'
                        self.data['SMA_50'].iloc[i],        # 'SMA_50'
                        self.data['RSI'].iloc[i],           # 'RSI'
                        self.data['Price_Change'].iloc[i],  # 'Price_Change'
                        self.data['Volume_Change'].iloc[i]  # 'Volume_Change'
                    ]

                    features = np.array(features).reshape(1, -1)  # Ensure features are 2D
                    scaled_features = self.scaler.transform(features)

                    # Make ML model prediction
                    ml_prediction = self.ml_model.predict(scaled_features)
                    if ml_prediction == 1:
                        print(f"ML-Confirmed Buy Signal at Fibonacci level {level:.2f} on {self.data.index[i]} at price {self.data['Close'].iloc[i]}")
                        self.signals.append((self.data.index[i], self.data['Close'].iloc[i], 'Buy'))
                    elif ml_prediction == 0:
                        print(f"ML-Confirmed Sell Signal at Fibonacci level {level:.2f} on {self.data.index[i]} at price {self.data['Close'].iloc[i]}")
                        self.signals.append((self.data.index[i], self.data['Close'].iloc[i], 'Sell'))
