from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

class ElliottWave:
    def __init__(self, data):
        self.data = data
        self.waves = []

    def detect_waves(self, lookback=5, distance=5):
        prices = self.data['Close'].values
        
        # Detect peaks (local maxima)
        peaks, _ = find_peaks(prices, distance=distance)
        # Detect troughs (local minima)
        troughs, _ = find_peaks(-prices, distance=distance)
        
        self.waves = sorted([(i, 'peak') for i in peaks] + [(i, 'trough') for i in troughs], key=lambda x: x[0])
        return self.waves

    def plot_waves(self):
        prices = self.data['Close']
        plt.plot(prices, label='Close Price')
        for idx, wave_type in self.waves:
            if wave_type == 'peak':
                plt.scatter(idx, prices[idx], color='green', marker='^', label='Peak')
            else:
                plt.scatter(idx, prices[idx], color='red', marker='v', label='Trough')
        plt.legend()
        plt.show()

    def label_waves(self):
        # Label 5-wave impulse and 3-wave corrective patterns
        impulse_count = 0
        corrective_count = 0

        for i, (idx, wave_type) in enumerate(self.waves):
            if impulse_count < 5:  # Label impulse waves
                self.waves[i] = (idx, wave_type, f'Impulse {impulse_count+1}')
                impulse_count += 1
            elif corrective_count < 3:  # Label corrective waves
                self.waves[i] = (idx, wave_type, f'Corrective {corrective_count+1}')
                corrective_count += 1
            if impulse_count == 5 and corrective_count == 3:
                impulse_count = 0
                corrective_count = 0

    def plot_labeled_waves(self):
        prices = self.data['Close']
        plt.plot(prices, label='Close Price')
        for idx, wave_type, label in self.waves:
            if wave_type == 'peak':
                plt.scatter(idx, prices[idx], color='green', marker='^', label=label)
            else:
                plt.scatter(idx, prices[idx], color='red', marker='v', label=label)
        plt.legend()
        plt.show()