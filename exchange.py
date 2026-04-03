"""
exchange.py — Exchange interface using CCXT for Binance.
Handles order execution, balance fetching, WebSocket candle streaming,
REST polling fallback, and paper trading simulation.
"""

import ccxt
import asyncio
import json
import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Callable, Any
from collections import defaultdict

import pandas as pd

from utils import ConfigLoader, Position


# ─────────────────────────────────────────────
#  EXCHANGE CLIENT (CCXT)
# ─────────────────────────────────────────────

class ExchangeClient:
    """
    Wrapper around CCXT for Binance.
    Supports: spot, swap (perpetual futures).
    Modes: live, testnet, paper.
    """

    def __init__(self, config: ConfigLoader, logger: logging.Logger, paper_mode: bool = False):
        self.config = config
        self.logger = logger
        self.paper_mode = paper_mode
        self.exchange = None
        self._public_exchange = None  # For data fetching in paper mode

        # Paper trading state
        self.paper_balance = config.get('backtest', 'initial_balance', default=10000.0)
        self.paper_orders: List[dict] = []
        self.paper_order_id = 1000

        if not paper_mode:
            self._init_exchange()
        else:
            self._init_public_exchange()

    def _init_exchange(self):
        """Initialize CCXT exchange instance."""
        exchange_name = self.config.get('exchange', 'name', default='binance').lower()
        api_key = self.config.get('exchange', 'api_key', default='')
        api_secret = self.config.get('exchange', 'api_secret', default='')
        api_password = self.config.get('exchange', 'api_password', default='')
        testnet = self.config.get('exchange', 'testnet', default=False)
        market_type = self.config.get('market_type', default='swap')

        exchange_class = getattr(ccxt, exchange_name)
        params = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': market_type,
                'adjustForTimeDifference': True,
            }
        }
        if api_password:
            params['password'] = api_password
            
        self.exchange = exchange_class(params)

        if testnet:
            self.exchange.set_sandbox_mode(True)
            self.logger.info(f"TESTNET mode enabled for {exchange_name}")
        else:
            self.logger.info(f"LIVE mode for {exchange_name}")

        # Test connectivity
        try:
            self.exchange.load_markets()
            self.logger.info(f"Exchange connected: {exchange_name} ({market_type})")
        except Exception as e:
            self.logger.error(f"Exchange connection failed: {e}")
            raise

    def _init_public_exchange(self):
        """Initialize a public (no auth) exchange for data fetching in paper mode."""
        exchange_name = self.config.get('exchange', 'name', default='binance').lower()
        market_type = self.config.get('market_type', default='swap')
        try:
            exchange_class = getattr(ccxt, exchange_name)
            self._public_exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': market_type,
                    'adjustForTimeDifference': True,
                }
            })
            self._public_exchange.load_markets()
            self.logger.info(f"Public exchange ({exchange_name}) initialized for paper trading data")
        except Exception as e:
            self.logger.warning(f"Public exchange init failed: {e}")
            self._public_exchange = None

    def _get_data_exchange(self):
        """Get the exchange instance to use for data fetching."""
        if self.exchange:
            return self.exchange
        if self._public_exchange:
            return self._public_exchange
        # Create on-demand
        try:
            self._public_exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            return self._public_exchange
        except Exception:
            return None

    def get_balance(self) -> float:
        """Get current USDT balance."""
        if self.paper_mode:
            return self.paper_balance
        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get('USDT', {})
            free = usdt.get('free', 0.0)
            self.logger.debug(f"Balance: {free:.2f} USDT")
            return float(free)
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return 0.0

    def get_ticker_price(self, symbol: str) -> float:
        """Get current ticker price."""
        try:
            ex = self._get_data_exchange()
            if ex:
                ticker = ex.fetch_ticker(symbol)
                return float(ticker['last'])
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}")
        return 0.0

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False
    ) -> Optional[dict]:
        """Place a market order."""
        if self.paper_mode:
            return self._paper_market_order(symbol, side, quantity)

        try:
            params = {}
            if reduce_only:
                params['reduceOnly'] = True

            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=quantity,
                params=params
            )
            self.logger.info(
                f"Market {side.upper()} order filled: {symbol} qty={quantity} "
                f"avg_price={order.get('average', 'N/A')}"
            )
            return order

        except ccxt.InsufficientFunds as e:
            self.logger.error(f"Insufficient funds: {e}")
        except ccxt.InvalidOrder as e:
            self.logger.error(f"Invalid order: {e}")
        except ccxt.RateLimitExceeded:
            self.logger.warning("Rate limit hit -- retrying in 2s...")
            time.sleep(2)
            return self.place_market_order(symbol, side, quantity, reduce_only)
        except Exception as e:
            self.logger.error(f"Order error: {e}")

        return None

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False
    ) -> Optional[dict]:
        """Place a limit order."""
        if self.paper_mode:
            return self._paper_limit_order(symbol, side, quantity, price)

        try:
            params = {}
            if reduce_only:
                params['reduceOnly'] = True

            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=quantity,
                price=price,
                params=params
            )
            self.logger.info(
                f"Limit {side.upper()} order placed: {symbol} qty={quantity} @ {price}"
            )
            return order

        except Exception as e:
            self.logger.error(f"Limit order error: {e}")
            return None

    def place_stop_loss(self, symbol: str, side: str, quantity: float, stop_price: float) -> Optional[dict]:
        """Place a stop-loss order (stop-market)."""
        if self.paper_mode:
            return {'id': f'paper_sl_{self.paper_order_id}', 'type': 'stop_loss'}

        try:
            sl_side = 'sell' if side == 'long' else 'buy'
            order = self.exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side=sl_side,
                amount=quantity,
                params={
                    'stopPrice': stop_price,
                    'reduceOnly': True,
                    'closePosition': False
                }
            )
            self.logger.info(f"SL order placed: {sl_side} {quantity} @ stop={stop_price}")
            return order
        except Exception as e:
            self.logger.error(f"SL order error: {e}")
            return None

    def place_take_profit(self, symbol: str, side: str, quantity: float, tp_price: float) -> Optional[dict]:
        """Place a take-profit order."""
        if self.paper_mode:
            return {'id': f'paper_tp_{self.paper_order_id}', 'type': 'take_profit'}

        try:
            tp_side = 'sell' if side == 'long' else 'buy'
            order = self.exchange.create_order(
                symbol=symbol,
                type='take_profit_market',
                side=tp_side,
                amount=quantity,
                params={
                    'stopPrice': tp_price,
                    'reduceOnly': True,
                    'closePosition': False
                }
            )
            self.logger.info(f"TP order placed: {tp_side} {quantity} @ stop={tp_price}")
            return order
        except Exception as e:
            self.logger.error(f"TP order error: {e}")
            return None

    def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol."""
        if self.paper_mode:
            self.paper_orders.clear()
            return
        try:
            self.exchange.cancel_all_orders(symbol)
            self.logger.info(f"All orders cancelled for {symbol}")
        except Exception as e:
            self.logger.error(f"Cancel orders error: {e}")

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        """Fetch recent OHLCV candles (works in all modes)."""
        try:
            ex = self._get_data_exchange()
            if ex is None:
                self.logger.error("No exchange available for data fetching")
                return pd.DataFrame()

            ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV: {e}")
            return pd.DataFrame()

    # ── Paper trading helpers ────────────────────

    def _paper_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        """Simulate a market order in paper mode."""
        self.paper_order_id += 1
        # Get current price for realistic fill
        price = self.get_ticker_price(symbol)
        order = {
            'id': f'paper_{self.paper_order_id}',
            'symbol': symbol,
            'side': side,
            'type': 'market',
            'amount': quantity,
            'average': price,
            'status': 'filled',
            'timestamp': int(time.time() * 1000)
        }
        self.logger.info(f"PAPER market {side.upper()}: {symbol} qty={quantity} @ {price:.2f}")
        return order

    def _paper_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        self.paper_order_id += 1
        order = {
            'id': f'paper_{self.paper_order_id}',
            'symbol': symbol,
            'side': side,
            'type': 'limit',
            'amount': quantity,
            'price': price,
            'average': price,
            'status': 'open',
            'timestamp': int(time.time() * 1000)
        }
        self.paper_orders.append(order)
        return order

    def update_paper_balance(self, pnl: float):
        """Update paper balance after a trade closes."""
        self.paper_balance += pnl


# ─────────────────────────────────────────────
#  WEBSOCKET CANDLE STREAMER (with REST fallback)
# ─────────────────────────────────────────────

class CandleStreamer:
    """
    Real-time candle streamer for Binance.
    Tries WebSocket first, falls back to REST polling if WS fails.
    """

    def __init__(
        self,
        symbol: str,
        on_new_5m_bar: Callable,
        on_new_1h_bar: Callable,
        logger: logging.Logger,
        testnet: bool = False,
        exchange_client: Optional['ExchangeClient'] = None
    ):
        self.raw_symbol = symbol
        self.symbol = symbol.replace('/', '').lower()
        self.on_new_5m_bar = on_new_5m_bar
        self.on_new_1h_bar = on_new_1h_bar
        self.logger = logger
        self.testnet = testnet
        self.exchange_client = exchange_client

        self.running = False
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60
        self._ws_failed_count = 0
        self._max_ws_failures = 3  # Switch to REST after N WS failures

        # Rolling candle buffers
        self.candles_5m: List[dict] = []
        self.candles_1h: List[dict] = []
        self.max_candles = 500

    def _get_ws_urls(self) -> List[str]:
        """Return a list of WebSocket URLs to try in order."""
        urls = []
        # Futures (XAUUSDT is on futures)
        urls.append(f"wss://fstream.binance.com/ws/{self.symbol}@kline_5m/{self.symbol}@kline_1h")
        # Spot
        urls.append(f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_5m/{self.symbol}@kline_1h")
        if self.testnet:
            urls.insert(0, f"wss://testnet.binance.vision/ws/{self.symbol}@kline_5m/{self.symbol}@kline_1h")
        return urls

    async def start(self):
        """Start streaming — tries WebSocket, falls back to REST polling."""
        self.running = True

        # Try WebSocket first
        ws_success = await self._try_websocket()

        # If WebSocket failed, use REST polling
        if self.running and not ws_success:
            self.logger.info("Switching to REST polling mode (every 30 seconds)...")
            await self._rest_polling_loop()

    async def _try_websocket(self) -> bool:
        """Attempt WebSocket connection with multiple URL fallbacks."""
        urls = self._get_ws_urls()

        for url in urls:
            if not self.running:
                return False
            try:
                import websockets
                self.logger.info(f"Trying WebSocket: {url}")
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self.logger.info("WebSocket connected!")
                    self._ws_failed_count = 0

                    async for message in ws:
                        if not self.running:
                            return True
                        await self._process_ws_message(message)

                    return True  # Clean disconnect

            except Exception as e:
                self.logger.warning(f"WebSocket failed ({url}): {e}")
                self._ws_failed_count += 1

        if self._ws_failed_count >= self._max_ws_failures:
            self.logger.warning(f"WebSocket failed {self._ws_failed_count} times. Using REST fallback.")
            return False
        return False

    async def _rest_polling_loop(self):
        """Fallback: poll REST API for new candles every 30 seconds."""
        last_5m_ts = None
        last_1h_ts = None
        poll_interval = 30  # seconds

        self.logger.info("REST polling active - checking for new candles every 30s")

        while self.running:
            try:
                ex = self.exchange_client._get_data_exchange() if self.exchange_client else None
                if ex is None:
                    exchange_name = 'binance'
                    if self.exchange_client and self.exchange_client.config:
                        exchange_name = self.exchange_client.config.get('exchange', 'name', default='binance').lower()
                    exchange_class = getattr(ccxt, exchange_name)
                    ex = exchange_class({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

                # Fetch latest 5M candles
                ohlcv_5m = ex.fetch_ohlcv(self.raw_symbol, '5m', limit=5)
                if ohlcv_5m:
                    for bar in ohlcv_5m:
                        ts = pd.Timestamp(bar[0], unit='ms', tz='UTC')
                        if last_5m_ts is None or ts > last_5m_ts:
                            candle = {
                                'timestamp': ts,
                                'open': float(bar[1]),
                                'high': float(bar[2]),
                                'low': float(bar[3]),
                                'close': float(bar[4]),
                                'volume': float(bar[5])
                            }
                            # Only process if this bar is CLOSED (not the current forming bar)
                            now = pd.Timestamp.now(tz='UTC')
                            bar_end = ts + pd.Timedelta(minutes=5)
                            if bar_end <= now:
                                self.candles_5m.append(candle)
                                if len(self.candles_5m) > self.max_candles:
                                    self.candles_5m = self.candles_5m[-self.max_candles:]
                                last_5m_ts = ts

                    # Trigger callback with updated data
                    if self.candles_5m:
                        await self.on_new_5m_bar(self._to_dataframe(self.candles_5m))

                # Fetch latest 1H candles
                ohlcv_1h = ex.fetch_ohlcv(self.raw_symbol, '1h', limit=3)
                if ohlcv_1h:
                    for bar in ohlcv_1h:
                        ts = pd.Timestamp(bar[0], unit='ms', tz='UTC')
                        if last_1h_ts is None or ts > last_1h_ts:
                            candle = {
                                'timestamp': ts,
                                'open': float(bar[1]),
                                'high': float(bar[2]),
                                'low': float(bar[3]),
                                'close': float(bar[4]),
                                'volume': float(bar[5])
                            }
                            now = pd.Timestamp.now(tz='UTC')
                            bar_end = ts + pd.Timedelta(hours=1)
                            if bar_end <= now:
                                self.candles_1h.append(candle)
                                if len(self.candles_1h) > self.max_candles:
                                    self.candles_1h = self.candles_1h[-self.max_candles:]
                                last_1h_ts = ts

                    if self.candles_1h:
                        await self.on_new_1h_bar(self._to_dataframe(self.candles_1h))

            except Exception as e:
                self.logger.error(f"REST polling error: {e}")

            # Wait before next poll
            for _ in range(poll_interval):
                if not self.running:
                    return
                await asyncio.sleep(1)

    async def _process_ws_message(self, message: str):
        """Process incoming WebSocket kline message."""
        try:
            data = json.loads(message)
            if 'k' not in data:
                return

            kline = data['k']
            interval = kline['i']
            is_closed = kline['x']

            candle = {
                'timestamp': pd.Timestamp(kline['t'], unit='ms', tz='UTC'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v'])
            }

            if interval == '5m' and is_closed:
                self.candles_5m.append(candle)
                if len(self.candles_5m) > self.max_candles:
                    self.candles_5m = self.candles_5m[-self.max_candles:]
                self.logger.debug(f"New 5M candle closed: {candle['close']}")
                await self.on_new_5m_bar(self._to_dataframe(self.candles_5m))

            elif interval == '1h' and is_closed:
                self.candles_1h.append(candle)
                if len(self.candles_1h) > self.max_candles:
                    self.candles_1h = self.candles_1h[-self.max_candles:]
                self.logger.debug(f"New 1H candle closed: {candle['close']}")
                await self.on_new_1h_bar(self._to_dataframe(self.candles_1h))

        except Exception as e:
            self.logger.error(f"Error processing WS message: {e}")

    def _to_dataframe(self, candles: List[dict]) -> pd.DataFrame:
        """Convert candle list to DataFrame."""
        if not candles:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return pd.DataFrame(candles)

    def stop(self):
        """Stop the streaming loop."""
        self.running = False
        self.logger.info("Candle streamer stopped")

    def initialize_historical(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame):
        """Pre-load historical candles for indicator warm-up."""
        for _, row in df_5m.iterrows():
            self.candles_5m.append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })
        for _, row in df_1h.iterrows():
            self.candles_1h.append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })
        self.logger.info(
            f"Historical data loaded: {len(self.candles_5m)} x 5M, {len(self.candles_1h)} x 1H candles"
        )
