from binance_data import BinanceData
from strategies.fibonacci import FibonacciRetracement
from strategies.breakout import BreakoutPatterns
from models.ml_model import MLModel
from logger import setup_logger

logger = setup_logger()


if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")
    
    if binance_data.data is not None:
        # Fibonacci Retracement
        fib_retracement = FibonacciRetracement(binance_data.data)
        fib_retracement.plot_levels()

        # Breakout Patterns
        breakout = BreakoutPatterns(binance_data.data)
        breakout.calculate_support_resistance()
        breakout.plot_support_resistance()

        # Machine Learning Model
        breakout_model = MLModel(binance_data.data)
        predictions, y_test = breakout_model.train_model()
        breakout_model.plot_predictions(predictions, y_test)




#--------------- ReversalPatterns
# from binance_data import BinanceData
# from strategies.reversal import ReversalPatterns
# from models.ml_model import MLReversalModel
# from logger import setup_logger


# if __name__ == '__main__':
#     # Fetch data from Binance
#     binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")
    
#     if binance_data.data is not None:
#         # Step 1: Detect Reversal Patterns
#         reversal = ReversalPatterns(binance_data.data)
#         reversal.detect_head_and_shoulders()
#         reversal.plot_patterns()

#         # Step 2: Train the Reversal Machine Learning model
#         reversal_model = MLReversalModel(binance_data.data)
#         predictions, y_test = reversal_model.train_model()
#         reversal_model.plot_predictions(predictions, y_test)




# Elliott Wave 1

# from binance_data import BinanceData
# from strategies.elliott_wave import ElliottWave
# from models.ml_model import MLElliottWaveModel
# from logger import setup_logger

# logger = setup_logger()

# if __name__ == '__main__':
#     # Fetch data from Binance
#     binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")
    
#     if binance_data.data is not None:
#         # Step 1: Detect Elliott Waves
#         elliott_wave = ElliottWave(binance_data.data)
#         elliott_wave.detect_waves()
#         elliott_wave.plot_waves()

#         # Step 2: Train the Elliott Wave ML model
#         wave_model = MLElliottWaveModel(binance_data.data)
#         predictions, y_test = wave_model.train_model()
#         wave_model.plot_predictions(predictions, y_test)



# Elliott Wave 2

# from binance_data import BinanceData
# from strategies.elliott_wave import ElliottWave
# from backtester import Backtester
# from logger import setup_logger

# logger = setup_logger()

# if __name__ == '__main__':
#     # Fetch data from Binance
#     binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")
    
#     if binance_data.data is not None:
#         # Step 1: Detect Elliott Waves and label them
#         elliott_wave = ElliottWave(binance_data.data)
#         elliott_wave.detect_waves()
#         elliott_wave.label_waves()

#         # Step 2: Backtest the strategy
#         backtester = Backtester(elliott_wave, binance_data.data)
#         backtester.run_backtest()


# Fair Value Gap (FVG)

# from binance_data import BinanceData
# from strategies.fair_value_gap import FairValueGap
# from models.ml_model import MLFVGModel
# from logger import setup_logger

# logger = setup_logger()

# if __name__ == '__main__':
#     # Fetch data from Binance
#     binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

#     if binance_data.data is not None:
#         # Step 1: Identify Fair Value Gaps
#         fvg = FairValueGap(binance_data.data)
#         fvg.identify_fvg()
#         fvg.plot_fvg()

#         # Step 2: Train the Fair Value Gap ML model
#         fvg_model = MLFVGModel(binance_data.data)
#         predictions, y_test = fvg_model.train_model()
#         fvg_model.plot_predictions(predictions, y_test)





# Candlestick Patterns

from binance_data import BinanceData
from strategies.candlestick_patterns import CandlestickPatterns
from models.ml_model import MLCandlestickModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Detect Candlestick Patterns
        candlestick_patterns = CandlestickPatterns(binance_data.data)
        candlestick_patterns.detect_doji()
        candlestick_patterns.detect_engulfing()
        candlestick_patterns.plot_patterns()

        # Step 2: Train the Candlestick Pattern ML model
        candle_model = MLCandlestickModel(binance_data.data)
        predictions, y_test = candle_model.train_model()
        candle_model.plot_predictions(predictions, y_test)




# Heikin Ashi

from binance_data import BinanceData
from strategies.heikin_ashi import HeikinAshi
from models.ml_model import MLHeikinAshiModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Calculate Heikin Ashi candles
        heikin_ashi = HeikinAshi(binance_data.data)
        heikin_ashi.calculate_heikin_ashi()
        heikin_ashi.plot_heikin_ashi()

        # Step 2: Train the Heikin Ashi ML model
        ha_model = MLHeikinAshiModel(binance_data.data, heikin_ashi.ha_data)
        predictions, y_test = ha_model.train_model()
        ha_model.plot_predictions(predictions, y_test)




# Moon Phases / Moon Cycles

from binance_data import BinanceData
from strategies.moon_phases import MoonPhases
from models.ml_model import MLMoonPhasesModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Generate and plot moon phases
        moon_phases = MoonPhases(binance_data.data)
        moon_phases.generate_mock_moon_phases()
        moon_phases.plot_moon_phases()

        # Step 2: Train the Moon Phases ML model
        moon_model = MLMoonPhasesModel(binance_data.data, moon_phases.moon_phases)
        predictions, y_test = moon_model.train_model()
        moon_model.plot_predictions(predictions, y_test)


# Renko Charts
# Renko Bricks Calculation: Simplifies price data by plotting bricks only when the price moves by a certain amount (e.g., $100).
# Trend Detection: Uses Renko bricks to identify uptrends and downtrends.
# Machine Learning Prediction: Predicts trend continuation or reversal using Renko brick patterns, RSI, MACD, and volume.

from binance_data import BinanceData
from strategies.renko import RenkoCharts
from models.ml_model import MLRenkoModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Calculate and plot Renko bricks
        renko_charts = RenkoCharts(binance_data.data, brick_size=100)
        renko_charts.calculate_renko()
        renko_charts.plot_renko()

        # Step 2: Train the Renko ML model
        renko_model = MLRenkoModel(binance_data.data, renko_charts.renko_data)
        predictions, y_test = renko_model.train_model()
        renko_model.plot_predictions(predictions, y_test)





# Harmonic Patterns

from binance_data import BinanceData
from strategies.harmonic_patterns import HarmonicPatterns
from models.ml_model import MLHarmonicModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Detect harmonic patterns (Gartley, Butterfly, Bat, Crab)
        harmonic_patterns = HarmonicPatterns(binance_data.data)
        harmonic_patterns.detect_gartley()
        harmonic_patterns.detect_butterfly()
        harmonic_patterns.detect_bat()
        harmonic_patterns.detect_crab()
        harmonic_patterns.plot_patterns()

        # Step 2: Train the Harmonic Patterns ML model
        harmonic_model = MLHarmonicModel(binance_data.data, harmonic_patterns.patterns)
        predictions, y_test = harmonic_model.train_model()
        harmonic_model.plot_predictions(predictions, y_test)





#  Support and Resistance

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




# Trend Lines

# from binance_data import BinanceData
# from strategies.trend_lines import TrendLines
# from models.ml_model import MLTrendLineModel
# from logger import setup_logger

# logger = setup_logger()

# if __name__ == '__main__':
#     # Fetch data from Binance
#     binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

#     if binance_data.data is not None:
#         # Step 1: Detect and plot trend lines
#         trend_lines = TrendLines(binance_data.data)
#         trend_lines.detect_trend_lines()
#         trend_lines.plot_trend_lines()

#         # Step 2: Train the ML model for trend line prediction
#         trend_model = MLTrendLineModel(binance_data.data, trend_lines.uptrend_lines, trend_lines.downtrend_lines)
#         predictions, y_test = trend_model.train_model()
#         trend_model.plot_predictions(predictions, y_test)

# v2

from binance_data import BinanceData
from strategies.trend_lines import TrendLines
from models.ml_model import MLTrendLineModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Detect multiple trend lines and breakouts
        trend_lines = TrendLines(binance_data.data)
        trend_lines.detect_trend_lines()
        trend_lines.detect_breakouts()
        trend_lines.plot_breakouts()

        # Step 2: Train the ML model for trend line prediction
        trend_model = MLTrendLineModel(binance_data.data, trend_lines.uptrend_lines, trend_lines.downtrend_lines)
        predictions, y_test = trend_model.train_model()
        trend_model.plot_predictions(predictions, y_test)




# Gann Fan / Gann Angles

from binance_data import BinanceData
from strategies.gann_fan import GannFan
from models.ml_model import MLGannModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Calculate and plot Gann angles
        gann_fan = GannFan(binance_data.data)
        gann_fan.calculate_gann_angles()
        gann_fan.plot_gann_fan()

        # Step 2: Train the Gann Fan ML model
        gann_model = MLGannModel(binance_data.data, gann_fan.angles)
        predictions, y_test = gann_model.train_model()
        gann_model.plot_predictions(predictions, y_test)






# Momentum Indicators

from binance_data import BinanceData
from strategies.momentum_indicators import MomentumIndicators
from models.ml_model import MLMomentumModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Calculate and plot momentum indicators
        momentum_indicators = MomentumIndicators(binance_data.data)
        momentum_indicators.plot_indicators()

        # Step 2: Train the Momentum Indicators ML model
        momentum_model = MLMomentumModel(binance_data.data)
        predictions, y_test = momentum_model.train_model()
        momentum_model.plot_predictions(predictions, y_test)







# Oscillators

from binance_data import BinanceData
from strategies.oscillators import Oscillators
from models.ml_model import MLOscillatorModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Calculate and plot oscillators
        oscillators = Oscillators(binance_data.data)
        oscillators.plot_oscillators()

        # Step 2: Train the Oscillator-based ML model
        oscillator_model = MLOscillatorModel(binance_data.data)
        predictions, y_test = oscillator_model.train_model()
        oscillator_model.plot_predictions(predictions, y_test)





# Divergence

from binance_data import BinanceData
from strategies.divergence import Divergence
from models.ml_model import MLDivergenceModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Detect and plot divergence
        divergence = Divergence(binance_data.data)
        divergence.detect_divergence(indicator='RSI')
        divergence.plot_divergence(indicator='RSI')

        # Step 2: Train the Divergence-based ML model
        divergence_model = MLDivergenceModel(binance_data.data, divergence.divergences)
        predictions, y_test = divergence_model.train_model()
        divergence_model.plot_predictions(predictions, y_test)







# Volume Indicators

from binance_data import BinanceData
from strategies.volume_indicators import VolumeIndicators
from models.ml_model import MLVolumeModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Calculate and plot volume indicators
        volume_indicators = VolumeIndicators(binance_data.data)
        volume_indicators.plot_volume_indicators()

        # Step 2: Train the Volume-based ML model
        volume_model = MLVolumeModel(binance_data.data)
        predictions, y_test = volume_model.train_model()
        volume_model.plot_predictions(predictions, y_test)





# Supply and Demand / Order Blocks

from binance_data import BinanceData
from strategies.supply_demand import SupplyDemand
from models.ml_model import MLSupplyDemandModel
from logger import setup_logger

logger = setup_logger()

if __name__ == '__main__':
    # Fetch data from Binance
    binance_data = BinanceData(symbol="BTCUSDT", interval="1d", start_str="1 Jan 2020")

    if binance_data.data is not None:
        # Step 1: Detect supply and demand zones
        supply_demand = SupplyDemand(binance_data.data)
        supply_demand.detect_zones()
        supply_demand.plot_zones()

        # Step 2: Train the Supply/Demand-based ML model
        supply_demand_model = MLSupplyDemandModel(binance_data.data, supply_demand.supply_zones, supply_demand.demand_zones)
        predictions, y_test = supply_demand_model.train_model()
        supply_demand_model.plot_predictions(predictions, y_test)





#  Market Structures