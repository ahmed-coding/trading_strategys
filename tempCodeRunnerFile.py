
from binance_data import BinanceData
from strategies.support_resistance import SupportResistance, DynamicSupportResistance
from models.ml_model import MLSupportResistanceModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Detect and plot static support/resistance levels
        support_resistance = SupportResistance(binance_data.data)
        support_resistance.detect_support_resistance()
        support_resistance.plot_support_resistance()

        # Step 2: Plot dynamic support and resistance using moving averages
        dynamic_sr = DynamicSupportResistance(binance_data.data)
        dynamic_sr.plot_moving_averages()

        # Step 3: Train the ML model for support/resistance prediction
        sr_model = MLSupportResistanceModel(binance_data.data, support_resistance.support_levels, support_resistance.resistance_levels)
        predictions, y_test = sr_model.train_model()
        sr_model.plot_predictions(predictions, y_test)

