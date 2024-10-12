from binance_data import BinanceData
from strategies.fibonacci import FibonacciRetracement
from strategies.breakout import BreakoutPatterns
from logger import setup_logger

logger = setup_logger()

class TradingBot:
    def __init__(self, strategy, symbol, interval, start_str):
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.start_str = start_str
        self.data = None

    def fetch_data(self):
        try:
            binance_data = BinanceData(self.symbol, self.interval, self.start_str)
            self.data = binance_data.data
            logger.info(f"Fetched data for {self.symbol}")
        except Exception as e:
            logger.error(f"Error fetching data: {e}")

    def execute_strategy(self):
        if self.data is not None:
            logger.info(f"Executing {self.strategy.__class__.__name__}")
            self.strategy.execute(self.data)
        else:
            logger.warning("No data available to execute the strategy.")

    def run(self):
        self.fetch_data()
        self.execute_strategy()

if __name__ == '__main__':
    # Example to use Fibonacci
    bot = TradingBot(FibonacciRetracement(), "BTCUSDT", "1d", "1 Jan 2020")
    bot.run()

    # Example to use Breakout
    bot = TradingBot(BreakoutPatterns(), "BTCUSDT", "1d", "1 Jan 2020")
    bot.run()
