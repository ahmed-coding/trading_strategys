from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import talib

class MLModel:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'Support': self.data['Low'].rolling(window=20).min(),
            'Resistance': self.data['High'].rolling(window=20).max(),
        }).dropna()

        features['Breakout'] = ((self.data['Close'] > features['Resistance']) | 
                                (self.data['Close'] < features['Support'])).shift(-1).dropna()

        X = features[['Close', 'Volume', 'Support', 'Resistance']]
        y = features['Breakout']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predictions')
        plt.plot(y_test.values, label='Actual')
        plt.legend()
        plt.show()


# 
class MLReversalModel:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'Support': self.data['Low'].rolling(window=20).min(),
            'Resistance': self.data['High'].rolling(window=20).max(),
        }).dropna()

        features['Reversal'] = ((self.data['Close'].pct_change() > 0.02) | 
                                (self.data['Close'].pct_change() < -0.02)).shift(-1).dropna()

        X = features[['Close', 'Volume', 'Support', 'Resistance']]
        y = features['Reversal']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predictions')
        plt.plot(y_test.values, label='Actual')
        plt.legend()
        plt.show()
        
        
        
        
class MLElliottWaveModel:
    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'MACD': talib.MACD(self.data['Close'])[0],
            'RSI': talib.RSI(self.data['Close']),
            'Bollinger_Upper': talib.BBANDS(self.data['Close'])[0],
            'Bollinger_Lower': talib.BBANDS(self.data['Close'])[2]
        }).dropna()

        features['Wave'] = (self.data['Close'].pct_change() > 0.02).shift(-1).dropna()

        X = features[['Close', 'Volume', 'MACD', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower']]
        y = features['Wave']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
    
    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Waves')
        plt.plot(y_test.values, label='Actual Waves')
        plt.legend()
        plt.show()
        


class MLFVGModel:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'RSI': talib.RSI(self.data['Close']),
            'MACD': talib.MACD(self.data['Close'])[0]
        }).dropna()

        # Label: 1 if price fills the gap within a specific period, 0 otherwise
        features['FVG_Filled'] = ((self.data['High'].shift(-5) > self.data['High']) & 
                                  (self.data['Low'].shift(-5) < self.data['Low'])).astype(int)

        X = features[['Close', 'Volume', 'RSI', 'MACD']]
        y = features['FVG_Filled']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predictions')
        plt.plot(y_test.values, label='Actual')
        plt.legend()
        plt.show()
        
class MLCandlestickModel:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'Doji': self.data['Doji'],
            'Bullish_Engulfing': self.data['Bullish_Engulfing'],
            'Bearish_Engulfing': self.data['Bearish_Engulfing']
        }).dropna()

        # Label: Predict upward or downward movement
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', 'Doji', 'Bullish_Engulfing', 'Bearish_Engulfing']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predictions')
        plt.plot(y_test.values, label='Actual Price Move')
        plt.legend()
        plt.show()
        
        
class MLHeikinAshiModel:
    def __init__(self, data, ha_data):
        self.data = data
        self.ha_data = ha_data
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'HA_Close': self.ha_data['HA_Close'],
            'HA_Open': self.ha_data['HA_Open'],
            'RSI': talib.RSI(self.data['Close']),
            'MACD': talib.MACD(self.data['Close'])[0]
        }).dropna()

        # Label: 1 if uptrend continues, 0 if downtrend
        features['Trend_Continuation'] = ((self.ha_data['HA_Close'].shift(-1) > self.ha_data['HA_Close'])).astype(int)

        X = features[['HA_Close', 'HA_Open', 'RSI', 'MACD']]
        y = features['Trend_Continuation']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()
        
        

        
class MLMoonPhasesModel:
    def __init__(self, data, moon_phases):
        self.data = data
        self.moon_phases = moon_phases
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'RSI': talib.RSI(self.data['Close']),
            'Moon_Phase': self.moon_phases['Phase']
        }).dropna()

        # Label: Predict upward or downward movement
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', 'RSI', 'Moon_Phase']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()
        
        
        
        
class MLRenkoModel:
    def __init__(self, data, renko_data):
        self.data = data
        self.renko_data = renko_data
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Renko_Price': self.renko_data['Price'],
            'RSI': talib.RSI(self.data['Close']),
            'MACD': talib.MACD(self.data['Close'])[0],
            'Volume': self.data['Volume']
        }).dropna()

        # Label: Predict if the trend continues or reverses (1 for trend continuation, 0 for reversal)
        features['Trend_Continuation'] = ((self.renko_data['Price'].shift(-1) > self.renko_data['Price']) | 
                                          (self.renko_data['Price'].shift(-1) < self.renko_data['Price'])).astype(int)

        X = features[['Renko_Price', 'RSI', 'MACD', 'Volume']]
        y = features['Trend_Continuation']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()




class MLHarmonicModel:
    def __init__(self, data, patterns):
        self.data = data
        self.patterns = patterns
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'RSI': talib.RSI(self.data['Close']),
            'MACD': talib.MACD(self.data['Close'])[0]
        }).dropna()

        # Label: Predict upward or downward movement after harmonic pattern detection
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        features['Harmonic_Pattern'] = features.index.isin([p[0] for p in self.patterns]).astype(int)

        X = features[['Close', 'Volume', 'RSI', 'MACD', 'Harmonic_Pattern']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()



class MLSupportResistanceModel:
    def __init__(self, data, support_levels, resistance_levels):
        self.data = data
        self.support_levels = support_levels
        self.resistance_levels = resistance_levels
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'RSI': talib.RSI(self.data['Close']),
            '50_MA': self.data['Close'].rolling(window=50).mean(),
            '200_MA': self.data['Close'].rolling(window=200).mean()
        }).dropna()

        # Label: Predict upward/downward movement at support/resistance levels
        features['At_Support'] = features.index.isin([s[0] for s in self.support_levels]).astype(int)
        features['At_Resistance'] = features.index.isin([r[0] for r in self.resistance_levels]).astype(int)
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', 'RSI', '50_MA', '200_MA', 'At_Support', 'At_Resistance']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Move')
        plt.plot(y_test.values, label='Actual Move')
        plt.legend()
        plt.show()



class MLTrendLineModel:
    def __init__(self, data, uptrend_lines, downtrend_lines):
        self.data = data
        self.uptrend_lines = uptrend_lines
        self.downtrend_lines = downtrend_lines
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'RSI': talib.RSI(self.data['Close']),
            '50_MA': self.data['Close'].rolling(window=50).mean(),
            '200_MA': self.data['Close'].rolling(window=200).mean()
        }).dropna()

        # Label: Predict upward or downward movement based on trend lines
        features['Uptrend_Line'] = features.index.isin([u[0] for u in self.uptrend_lines]).astype(int)
        features['Downtrend_Line'] = features.index.isin([d[0] for d in self.downtrend_lines]).astype(int)
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', 'RSI', '50_MA', '200_MA', 'Uptrend_Line', 'Downtrend_Line']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()



class MLGannModel:
    def __init__(self, data, gann_angles):
        self.data = data
        self.gann_angles = gann_angles
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'RSI': talib.RSI(self.data['Close']),
            'MACD': talib.MACD(self.data['Close'])[0],
            '1x1': [angle['1x1'] for angle in self.gann_angles],
            '2x1': [angle['2x1'] for angle in self.gann_angles],
            '3x1': [angle['3x1'] for angle in self.gann_angles]
        }).dropna()

        # Label: Predict upward or downward movement
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', 'RSI', 'MACD', '1x1', '2x1', '3x1']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()




class MLMomentumModel:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'RSI': self.data['RSI'],
            'MACD': self.data['MACD'],
            'Stochastic_K': self.data['Stochastic_K'],
            'Stochastic_D': self.data['Stochastic_D']
        }).dropna()

        # Label: Predict upward or downward movement
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', 'RSI', 'MACD', 'Stochastic_K', 'Stochastic_D']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()




class MLOscillatorModel:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'RSI': self.data['RSI'],
            'Stochastic_K': self.data['Stochastic_K'],
            'Stochastic_D': self.data['Stochastic_D'],
            'CCI': self.data['CCI']
        }).dropna()

        # Label: Predict upward or downward movement
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', 'RSI', 'Stochastic_K', 'Stochastic_D', 'CCI']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()



class MLSupplyDemandModel:
    def __init__(self, data, supply_zones, demand_zones):
        self.data = data
        self.supply_zones = supply_zones
        self.demand_zones = demand_zones
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            '50_MA': self.data['Close'].rolling(window=50).mean(),
            '200_MA': self.data['Close'].rolling(window=200).mean()
        }).dropna()

        # Add supply and demand zones as features
        features['Supply_Zone'] = features.index.isin([s[0] for s in self.supply_zones]).astype(int)
        features['Demand_Zone'] = features.index.isin([d[0] for d in self.demand_zones]).astype(int)

        # Label: Predict upward or downward movement
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', '50_MA', '200_MA', 'Supply_Zone', 'Demand_Zone']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()




class MLMarketStructureModel:
    def __init__(self, data, highs, lows, bos):
        self.data = data
        self.highs = highs
        self.lows = lows
        self.bos = bos
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            '50_MA': self.data['Close'].rolling(window=50).mean(),
            '200_MA': self.data['Close'].rolling(window=200).mean()
        }).dropna()

        # Add market structure and BOS as features
        features['Highs'] = features.index.isin([h[0] for h in self.highs]).astype(int)
        features['Lows'] = features.index.isin([l[0] for l in self.lows]).astype(int)
        features['Break_of_Structure'] = features.index.isin([b[0] for b in self.bos]).astype(int)

        # Label: Predict upward or downward movement
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', '50_MA', '200_MA', 'Highs', 'Lows', 'Break_of_Structure']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()



class MLBOSModel:
    def __init__(self, data, bos):
        self.data = data
        self.bos = bos
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            '50_MA': self.data['Close'].rolling(window=50).mean(),
            '200_MA': self.data['Close'].rolling(window=200).mean()
        }).dropna()

        # Add BOS as a feature
        features['Break_of_Structure'] = features.index.isin([b[0] for b in self.bos]).astype(int)

        # Label: Predict upward or downward movement
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', '50_MA', '200_MA', 'Break_of_Structure']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()







class MLCHoCHModel:
    def __init__(self, data, choch_signals):
        self.data = data
        self.choch_signals = choch_signals
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            '50_MA': self.data['Close'].rolling(window=50).mean(),
            '200_MA': self.data['Close'].rolling(window=200).mean()
        }).dropna()

        # Add CHoCH as a feature
        features['CHoCH'] = features.index.isin([s[0] for s in self.choch_signals]).astype(int)

        # Label: Predict upward or downward movement
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', '50_MA', '200_MA', 'CHoCH']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()






class MLDivergenceModel:
    def __init__(self, data, divergences):
        self.data = data
        self.divergences = divergences
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'RSI': self.data['RSI'],
            'MACD': self.data['MACD'],
            'Stochastic_K': self.data['Stochastic_K']
        }).dropna()

        # Label: Detect upward/downward movement
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        features['Divergence'] = features.index.isin([d[0] for d in self.divergences]).astype(int)

        X = features[['Close', 'Volume', 'RSI', 'MACD', 'Stochastic_K', 'Divergence']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()



class MLVolumeModel:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def prepare_data(self):
        features = pd.DataFrame({
            'Close': self.data['Close'],
            'Volume': self.data['Volume'],
            'OBV': self.data['OBV'],
            'CMF': self.data['CMF'],
            'VWAP': self.data['VWAP']
        }).dropna()

        # Label: Predict upward or downward movement
        features['Price_Move'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)

        X = features[['Close', 'Volume', 'OBV', 'CMF', 'VWAP']]
        y = features['Price_Move']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        print(f"Accuracy: {accuracy}")
        return predictions, y_test

    def plot_predictions(self, predictions, y_test):
        plt.plot(predictions, label='Predicted Trend')
        plt.plot(y_test.values, label='Actual Trend')
        plt.legend()
        plt.show()
