import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Summary:
# Gann Angle Calculation: Calculates Gann angles (1x1, 2x1, 3x1, etc.) to analyze how price moves in relation to time.
# Gann Fan Plotting: Draws Gann angles on the price chart to visualize potential support and resistance levels.
# Machine Learning Prediction: Uses Gann angles and other technical indicators (RSI, MACD) to predict future price movements.


class GannFan:
    def __init__(self, data):
        self.data = data
        self.angles = []

    def calculate_gann_angles(self):
        """
        Calculate Gann angles from specific price points.
        1x1 represents 45-degree line. Other angles are variations.
        """
        starting_price = self.data['Close'][0]
        for i in range(1, len(self.data)):
            time_diff = i  # Time difference from starting point
            self.angles.append({
                '1x1': starting_price + time_diff * 1,
                '2x1': starting_price + time_diff * 2,
                '3x1': starting_price + time_diff * 3,
                '1x2': starting_price + time_diff / 2,
            })

    def plot_gann_fan(self):
        """
        Plot the Gann fan angles (1x1, 2x1, etc.) on the chart.
        """
        plt.plot(self.data['Close'], label='Close Price')

        for i, angle in enumerate(self.angles):
            plt.plot(i, angle['1x1'], color='blue', linestyle='--', label='1x1 Angle' if i == 1 else "")
            plt.plot(i, angle['2x1'], color='green', linestyle='--', label='2x1 Angle' if i == 1 else "")
            plt.plot(i, angle['3x1'], color='red', linestyle='--', label='3x1 Angle' if i == 1 else "")
            plt.plot(i, angle['1x2'], color='purple', linestyle='--', label='1x2 Angle' if i == 1 else "")

        plt.legend()
        plt.show()
