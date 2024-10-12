import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Summary:
# Renko Bricks Calculation: Simplifies price data by plotting bricks only when the price moves by a certain amount (e.g., $100).
# Trend Detection: Uses Renko bricks to identify uptrends and downtrends.
# Machine Learning Prediction: Predicts trend continuation or reversal using Renko brick patterns, RSI, MACD, and volume.


class RenkoCharts:
    def __init__(self, data, brick_size=100):
        self.data = data
        self.brick_size = brick_size
        self.renko_data = pd.DataFrame()

    def calculate_renko(self):
        prices = self.data['Close'].values
        bricks = []
        brick_direction = 0  # 1 for uptrend, -1 for downtrend
        current_brick = prices[0]

        for price in prices:
            if price >= current_brick + self.brick_size:
                bricks.append(current_brick + self.brick_size)
                current_brick += self.brick_size
                brick_direction = 1
            elif price <= current_brick - self.brick_size:
                bricks.append(current_brick - self.brick_size)
                current_brick -= self.brick_size
                brick_direction = -1

        self.renko_data = pd.DataFrame({'Price': bricks, 'Direction': [brick_direction] * len(bricks)})

    def plot_renko(self):
        plt.plot(self.data['Close'], label='Close Price')
        plt.step(self.renko_data.index, self.renko_data['Price'], where='post', label='Renko Bricks', color='red')
        plt.fill_between(self.renko_data.index, self.renko_data['Price'], color='gray', step='post', alpha=0.3)
        plt.legend()
        plt.show()
