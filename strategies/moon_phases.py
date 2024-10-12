import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MoonPhases:
    def __init__(self, data):
        self.data = data
        self.moon_phases = pd.DataFrame(index=self.data.index)

    def generate_mock_moon_phases(self):
        """
        Generate mock moon phases (Full Moon = 1, New Moon = -1) for demonstration purposes.
        In a real implementation, you would fetch moon phase data from an external source.
        """
        self.moon_phases['Phase'] = np.sin(np.linspace(0, 4 * np.pi, len(self.data)))  # Simulates lunar cycle
        self.moon_phases['Phase'] = np.where(self.moon_phases['Phase'] > 0, 1, -1)  # 1 = Full Moon, -1 = New Moon

    def plot_moon_phases(self):
        plt.plot(self.data['Close'], label='Close Price')
        plt.scatter(self.moon_phases.index, self.data['Close'][self.moon_phases['Phase'] == 1], color='blue', label='Full Moon', marker='o')
        plt.scatter(self.moon_phases.index, self.data['Close'][self.moon_phases['Phase'] == -1], color='red', label='New Moon', marker='x')
        plt.legend()
        plt.show()
