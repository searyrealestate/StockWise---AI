# realtime_data_feed.py
import asyncio
import os
import pandas as pd
import logging
import streamlit as st

try:
    # --- NEW: Use the modern alpaca-sdk ---
    from alpaca_sdk.streams import StockDataStream
except ImportError:
    print("Alpaca SDK not installed. Run: pip install alpaca-py")

    # Create dummy classes to avoid crashing on import
    class Stream:
        pass


    class URL:
        pass

# --- Configuration ---
# Store these securely in st.secrets or environment variables
API_KEY = st.secrets.get('APCA_API_KEY_ID')
API_SECRET = st.secrets.get('APCA_API_SECRET_KEY')


logger = logging.getLogger(__name__)


class RealTimeDataFeed:
    """
    Manages a real-time websocket connection for stock data
    and provides a buffered, asynchronous queue for consumption.
    """

    def __init__(self, symbols: list):
        self.symbols = symbols
        self._is_running = False

        if not API_KEY or not API_SECRET:
            logger.error("Alpaca API Key/Secret not found. Real-time feed disabled.")
            self.stream = None
        else:
            # --- Use alpaca_sdk syntax ---
            self.stream = StockDataStream(
                api_key=API_KEY,
                secret_key=API_SECRET,
                paper=True
            )

        # The asyncio.Queue is the "buffer" you requested
        self.trade_queue = asyncio.Queue()
        self.last_price = {symbol: None for symbol in symbols}
        self.price_window = {symbol: [] for symbol in symbols}
        self.window_size = 20  # For outlier detection

    async def _trade_handler(self, trade):
        """
        Async callback for every new trade received from the websocket.
        """
        symbol = trade.symbol
        price = trade.price

        # --- 1. Outlier Filtering (as you requested) ---
        if self.last_price[symbol]:
            # Simple std-dev based filter
            if len(self.price_window[symbol]) >= self.window_size:
                window = pd.Series(self.price_window[symbol])
                mean = window.mean()
                std = window.std()

                # If price is more than 3 std-devs from the mean, it's a potential outlier
                if std > 0 and abs(price - mean) > (3 * std):
                    logger.warning(f"OUTLIER DETECTED for {symbol}: Price {price}. Mean: {mean}, Std: {std}")
                    return  # Skip this trade

            # Update the rolling window
            self.price_window[symbol].append(price)
            if len(self.price_window[symbol]) > self.window_size:
                self.price_window[symbol].pop(0)

        # --- 2. Preprocessing & Buffering ---
        self.last_price[symbol] = price
        data_packet = {
            'type': 'trade',
            'symbol': symbol,
            'price': price,
            'timestamp': pd.to_datetime(trade.timestamp, utc=True)
        }

        # Put the cleaned data into the async queue
        await self.trade_queue.put(data_packet)

    async def run(self):
        """
        Starts the websocket connection and registers the handler.
        """
        if not self.stream:
            logger.error("Cannot run feed: Stream is not initialized (check API keys).")
            return

        if self._is_running:
            logger.warning("Feed is already running.")
            return

        self._is_running = True
        logger.info(f"Subscribing to real-time trades for: {self.symbols}")
        try:
            self.stream.subscribe_trades(self._trade_handler, *self.symbols)
            await self.stream.run()
        except Exception as e:
            logger.error(f"Error in data feed run(): {e}", exc_info=True)
        finally:
            self._is_running = False
            logger.info("Real-time data feed stopped.")

    async def stop(self):
        """
        Stops the websocket connection.
        """
        if self.stream and self._is_running:
            logger.info("Stopping real-time data feed...")
            await self.stream.stop()
            self._is_running = False


# --- Example of how to run/test this ---
async def main_app_loop(feed: RealTimeDataFeed):
    """
    This simulates how your main app would consume data.
    """
    logger.info("Starting main app loop, waiting for data from queue...")
    while True:
        try:
            # Wait for the next item from the feed,
            # allowing other async tasks to run.
            data = await asyncio.wait_for(feed.trade_queue.get(), timeout=5.0)

            # --- HERE: Trigger your ML models/analysis ---
            logger.info(f"APP RECEIVED: {data['symbol']} @ {data['price']}")
            # e.g., await professional_advisor.on_new_data(data)

            feed.trade_queue.task_done()
        except asyncio.TimeoutError:
            # No data in 5 seconds, just loop again.
            pass
        except Exception as e:
            logger.error(f"Error in main app loop: {e}", exc_info=True)
            break


if __name__ == "__main__":
    # To run this test, you would need to:
    # 1. Set your APCA_API_KEY_ID and APCA_API_SECRET_KEY as environment variables
    # 2. Run this file: python realtime_data_feed.py

    # Setup basic logging for the test
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    symbols_to_watch = ['NVDA', 'AAPL', 'MSFT']
    data_feed = RealTimeDataFeed(symbols=symbols_to_watch)


    async def start_services():
        # Run the data feed and the main app loop concurrently
        feed_task = asyncio.create_task(data_feed.run())
        app_task = asyncio.create_task(main_app_loop(data_feed))
        await asyncio.gather(feed_task, app_task)


    try:
        asyncio.run(start_services())
    except KeyboardInterrupt:
        logger.info("Shutting down...")