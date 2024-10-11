import pandas as pd

class CandlestickPatterns:
    def __init__(self, data):
        self.data = data

    def detect_doji(self):
        """
        Doji candlestick pattern: A candlestick where the open and close prices are very close to each other.
        """
        self.data['Doji'] = (abs(self.data['Close'] - self.data['Open']) / (self.data['High'] - self.data['Low']) < 0.1).astype(int)

    def detect_engulfing(self):
        """
        Engulfing pattern: Bullish or bearish based on whether the body of the current candlestick fully engulfs the previous one.
        """
        prev = self.data.shift(1)
        self.data['Bullish_Engulfing'] = ((self.data['Close'] > self.data['Open']) & 
                                          (prev['Close'] < prev['Open']) & 
                                          (self.data['Close'] > prev['Open']) & 
                                          (self.data['Open'] < prev['Close'])).astype(int)
        self.data['Bearish_Engulfing'] = ((self.data['Close'] < self.data['Open']) & 
                                          (prev['Close'] > prev['Open']) & 
                                          (self.data['Close'] < prev['Open']) & 
                                          (self.data['Open'] > prev['Close'])).astype(int)

    def plot_patterns(self):
        """
        Plot the detected candlestick patterns on the price chart.
        """
        # Plot the price chart
        self.data['Close'].plot(label='Close Price')

        # Highlight Doji and Engulfing patterns
        doji_indices = self.data[self.data['Doji'] == 1].index
        engulfing_indices = self.data[(self.data['Bullish_Engulfing'] == 1) | (self.data['Bearish_Engulfing'] == 1)].index

        plt.scatter(doji_indices, self.data['Close'][doji_indices], color='yellow', label='Doji', marker='o')
        plt.scatter(engulfing_indices, self.data['Close'][engulfing_indices], color='green', label='Engulfing', marker='x')

        plt.legend()
        plt.show()
