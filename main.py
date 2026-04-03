"""
main.py — Gold Momentum Multi-TF Scalper Bot entry point.

Modes:
  1. backtest  — Run historical simulation on CSV/Parquet data
  2. paper     — Paper trading with real-time WebSocket data
  3. live      — Live trading (testnet or real) on Binance

Usage:
  python main.py --mode backtest --data data/XAUUSDT_5m.csv
  python main.py --mode paper
  python main.py --mode live
  python main.py --mode download --since "2024-01-01T00:00:00Z" --output data/XAUUSDT_5m.csv
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import signal
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import pandas as pd

from utils import (
    ConfigLoader, DatabaseManager, TelegramAlerter, NewsFilter, DXYFilter,
    Position, TradeRecord,
    setup_logging, is_trading_session, calculate_position_size, generate_trade_id
)
from strategy import StrategyEngine, Signal
from backtester import Backtester, DataLoader, BacktestResult
from exchange import ExchangeClient, CandleStreamer


# ─────────────────────────────────────────────
#  LIVE TRADING BOT
# ─────────────────────────────────────────────

class TradingBot:
    """
    Main trading bot orchestrator.
    Manages positions, processes signals, handles execution.
    """

    def __init__(self, config: ConfigLoader, mode: str = 'paper'):
        self.config = config
        self.mode = mode  # 'paper' or 'live'
        self.logger = setup_logging(config)
        self.logger.info(f"🚀 Gold Momentum Multi-TF Scalper — Mode: {mode.upper()}")

        # Components
        self.db = DatabaseManager(config.get('database', 'path', default='trades.db'))
        self.telegram = TelegramAlerter(config, self.logger)
        self.news_filter = NewsFilter(self.logger)
        self.dxy_filter = DXYFilter(enabled=False, logger=self.logger)

        # Exchange
        is_paper = (mode == 'paper')
        self.exchange = ExchangeClient(config, self.logger, paper_mode=is_paper)

        # Strategy
        self.strategy = StrategyEngine(config, self.logger)

        # State
        self.positions: List[Position] = []
        self.symbol = config['symbol']
        self.risk_pct = config.get('risk', 'risk_pct', default=0.5)
        self.max_positions = config.get('risk', 'max_concurrent_positions', default=2)
        self.max_daily_dd = config.get('risk', 'max_daily_drawdown_pct', default=5.0)
        self.order_type = config.get('execution', 'order_type', default='market')

        # Daily tracking
        self.day_start_balance = 0.0
        self.current_day = ''
        self.emergency_stop = False

        # DataFrames for indicators (maintained via WebSocket)
        self.df_5m: Optional[pd.DataFrame] = None
        self.df_1h: Optional[pd.DataFrame] = None

        self.running = False

    async def start(self):
        """Start the live/paper trading bot."""
        self.running = True
        self.logger.info("=" * 60)
        self.logger.info("  STARTING TRADING BOT")
        self.logger.info("=" * 60)

        # ── Fetch initial historical data for indicator warm-up ──
        self.logger.info("Fetching historical data for indicator warm-up...")
        self.df_5m = self.exchange.fetch_ohlcv(self.symbol, '5m', limit=300)
        self.df_1h = self.exchange.fetch_ohlcv(self.symbol, '1h', limit=300)

        if self.df_5m.empty or self.df_1h.empty:
            self.logger.error("Failed to fetch historical data. Exiting.")
            return

        # Compute indicators on initial data
        self.df_5m = self.strategy.compute_indicators(self.df_5m)
        self.df_1h = self.strategy.compute_indicators(self.df_1h)

        self.logger.info(f"Loaded {len(self.df_5m)} x 5M and {len(self.df_1h)} x 1H candles")

        # ── Get initial balance ──
        balance = self.exchange.get_balance()
        self.day_start_balance = balance
        self.current_day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self.logger.info(f"💰 Starting balance: ${balance:,.2f} USDT")

        # ── Setup candle streamer (WebSocket + REST fallback) ──
        testnet = self.config.get('exchange', 'testnet', default=True)
        streamer = CandleStreamer(
            symbol=self.symbol,
            on_new_5m_bar=self._on_new_5m_bar,
            on_new_1h_bar=self._on_new_1h_bar,
            logger=self.logger,
            testnet=testnet,
            exchange_client=self.exchange
        )

        # Pre-load historical data into streamer
        streamer.initialize_historical(self.df_5m, self.df_1h)

        # ── Send startup alert ──
        self.telegram.send_alert(
            f"🚀 Bot started in {self.mode.upper()} mode\n"
            f"Symbol: {self.symbol}\n"
            f"Balance: ${balance:,.2f}\n"
            f"Risk: {self.risk_pct}%"
        )

        # ── Run WebSocket ──
        try:
            await streamer.start()
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user.")
        finally:
            streamer.stop()
            self.db.close()
            self.logger.info("Bot shutdown complete.")

    async def _on_new_5m_bar(self, df_5m_raw: pd.DataFrame):
        """
        Called when a new 5M candle closes.
        This is the core trading logic loop.
        """
        if not self.running or self.emergency_stop:
            return

        try:
            # ── Recompute indicators ──
            self.df_5m = self.strategy.compute_indicators(df_5m_raw)
            utc_now = datetime.now(timezone.utc)

            # ── Daily drawdown check ──
            balance = self.exchange.get_balance()
            today = utc_now.strftime('%Y-%m-%d')
            if today != self.current_day:
                self.current_day = today
                self.day_start_balance = balance
                self.emergency_stop = False

            if self.day_start_balance > 0:
                daily_dd = (self.day_start_balance - balance) / self.day_start_balance * 100
                if daily_dd >= self.max_daily_dd:
                    self.logger.critical(
                        f"🚨 EMERGENCY STOP! Daily drawdown {daily_dd:.2f}% >= {self.max_daily_dd}%. "
                        f"Closing all positions."
                    )
                    self.emergency_stop = True
                    await self._close_all_positions("emergency_dd_stop")
                    self.telegram.send_alert(
                        f"🚨 EMERGENCY STOP!\nDaily DD: {daily_dd:.2f}%\nAll positions closed."
                    )
                    return

            # ── News filter ──
            if self.news_filter.should_skip_trade(utc_now):
                self.logger.info("⏸ Skipping — high-impact news window.")
                return

            # ── Check exits on existing positions ──
            if self.df_5m is not None and not self.df_5m.empty:
                last_bar = self.df_5m.iloc[-1]
                atr = last_bar['atr'] if not pd.isna(last_bar['atr']) else 0
                current_price = last_bar['close']
                current_high = last_bar['high']
                current_low = last_bar['low']

                for pos in list(self.positions):
                    exit_reason, exit_price, close_qty = self.strategy.check_exit_conditions(
                        pos, current_price, current_high, current_low, atr
                    )

                    if exit_reason:
                        await self._execute_exit(pos, exit_reason, exit_price, close_qty, balance)

            # ── Evaluate new entry signal ──
            if self.df_1h is not None and not self.df_1h.empty:
                signal = self.strategy.evaluate(
                    self.df_1h, self.df_5m, self.positions, utc_now
                )

                if signal:
                    # DXY filter
                    if self.dxy_filter.should_skip(signal.side):
                        return

                    await self._execute_entry(signal, balance)

        except Exception as e:
            self.logger.error(f"Error in 5M bar handler: {e}", exc_info=True)

    async def _on_new_1h_bar(self, df_1h_raw: pd.DataFrame):
        """Called when a new 1H candle closes — update 1H data."""
        try:
            self.df_1h = self.strategy.compute_indicators(df_1h_raw)
            self.logger.info(f"📊 1H candle updated — {len(self.df_1h)} bars")
        except Exception as e:
            self.logger.error(f"Error in 1H bar handler: {e}", exc_info=True)

    async def _execute_entry(self, signal: Signal, balance: float):
        """Execute a new trade entry."""
        try:
            # Calculate position size
            qty = calculate_position_size(
                balance, self.risk_pct, signal.entry_price, signal.sl_price
            )

            if qty <= 0:
                self.logger.warning("Position size too small — skipping signal.")
                return

            # Place order
            order_side = 'buy' if signal.side == 'long' else 'sell'

            if self.order_type == 'market':
                order = self.exchange.place_market_order(self.symbol, order_side, qty)
            else:
                order = self.exchange.place_limit_order(
                    self.symbol, order_side, qty, signal.entry_price
                )

            if order is None:
                self.logger.error("Order execution failed — skipping.")
                return

            # Get fill price
            fill_price = float(order.get('average', signal.entry_price))

            # Create position
            trade_id = generate_trade_id(signal.side)
            new_pos = Position(
                trade_id=trade_id,
                symbol=self.symbol,
                side=signal.side,
                entry_price=fill_price,
                entry_time=datetime.now(timezone.utc),
                quantity=qty,
                remaining_qty=qty,
                sl_price=signal.sl_price,
                tp1_price=signal.tp1_price,
                atr_at_entry=signal.atr_value,
                highest_since_entry=fill_price if signal.side == 'long' else 0,
                lowest_since_entry=fill_price if signal.side == 'short' else 999999
            )
            self.positions.append(new_pos)

            # Place SL/TP orders on exchange (if not paper)
            if not self.exchange.paper_mode:
                self.exchange.place_stop_loss(self.symbol, signal.side, qty, signal.sl_price)
                self.exchange.place_take_profit(
                    self.symbol, signal.side,
                    round(qty * self.strategy.tp1_close_pct, 8),
                    signal.tp1_price
                )

            # Log to DB
            trade_record = TradeRecord(
                trade_id=trade_id,
                symbol=self.symbol,
                side=signal.side,
                entry_price=fill_price,
                entry_time=datetime.now(timezone.utc).isoformat(),
                quantity=qty,
                sl_price=signal.sl_price,
                tp1_price=signal.tp1_price,
                entry_reason=signal.reason,
                remaining_qty=qty
            )
            self.db.insert_trade(trade_record)

            # Telegram
            self.telegram.send_trade_open(trade_record)

            self.logger.info(
                f"✅ ENTRY {signal.side.upper()} | Price: {fill_price:.2f} | "
                f"SL: {signal.sl_price:.2f} | TP1: {signal.tp1_price:.2f} | "
                f"Qty: {qty:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Entry execution error: {e}", exc_info=True)

    async def _execute_exit(
        self, pos: Position, reason: str,
        exit_price: float, close_qty: float, balance: float
    ):
        """Execute a trade exit (full or partial)."""
        try:
            # Place close order
            close_side = 'sell' if pos.side == 'long' else 'buy'

            if self.order_type == 'market':
                order = self.exchange.place_market_order(
                    self.symbol, close_side, close_qty, reduce_only=True
                )
            else:
                order = self.exchange.place_limit_order(
                    self.symbol, close_side, close_qty, exit_price, reduce_only=True
                )

            fill_price = float(order.get('average', exit_price)) if order else exit_price

            # Calculate PnL
            if pos.side == 'long':
                pnl = (fill_price - pos.entry_price) * close_qty
            else:
                pnl = (pos.entry_price - fill_price) * close_qty

            if reason == 'tp1':
                # Partial close
                pos.remaining_qty -= close_qty
                pos.tp1_hit = True
                self.logger.info(
                    f"  TP1 partial close: {close_qty:.4f} @ {fill_price:.2f} | PnL: {pnl:+.2f}"
                )
                # Cancel existing SL/TP, set new trailing stop
                if not self.exchange.paper_mode:
                    self.exchange.cancel_all_orders(self.symbol)
                    # Place new SL at trail level
                    atr = pos.atr_at_entry
                    if pos.side == 'long':
                        pos.highest_since_entry = fill_price
                        pos.trail_stop = fill_price - (self.strategy.trail_atr_mult * atr)
                    else:
                        pos.lowest_since_entry = fill_price
                        pos.trail_stop = fill_price + (self.strategy.trail_atr_mult * atr)
                    self.exchange.place_stop_loss(
                        self.symbol, pos.side, pos.remaining_qty, pos.trail_stop
                    )
            else:
                # Full close
                if self.exchange.paper_mode:
                    self.exchange.update_paper_balance(pnl)

                pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100

                self.db.update_trade(pos.trade_id, {
                    'status': 'closed',
                    'exit_price': fill_price,
                    'exit_time': datetime.now(timezone.utc).isoformat(),
                    'pnl': round(pnl, 4),
                    'pnl_pct': round(pnl_pct, 4),
                    'exit_reason': reason,
                    'remaining_qty': 0
                })

                # Update daily PnL
                today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                self.db.update_daily_pnl(today, pnl, pnl > 0, balance + pnl)

                # Telegram
                trade_rec = TradeRecord(
                    trade_id=pos.trade_id,
                    symbol=pos.symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    entry_time=pos.entry_time.isoformat(),
                    quantity=pos.quantity,
                    sl_price=pos.sl_price,
                    tp1_price=pos.tp1_price,
                    status='closed',
                    exit_price=fill_price,
                    exit_time=datetime.now(timezone.utc).isoformat(),
                    pnl=round(pnl, 4),
                    pnl_pct=round(pnl_pct, 4),
                    exit_reason=reason
                )
                self.telegram.send_trade_close(trade_rec)

                self.positions.remove(pos)

                if not self.exchange.paper_mode:
                    self.exchange.cancel_all_orders(self.symbol)

                self.logger.info(
                    f"  CLOSED {pos.side.upper()} @ {fill_price:.2f} | "
                    f"Reason: {reason} | PnL: {pnl:+.2f} ({pnl_pct:+.2f}%)"
                )

        except Exception as e:
            self.logger.error(f"Exit execution error: {e}", exc_info=True)

    async def _close_all_positions(self, reason: str):
        """Emergency: close all open positions."""
        for pos in list(self.positions):
            close_side = 'sell' if pos.side == 'long' else 'buy'
            self.exchange.place_market_order(
                self.symbol, close_side, pos.remaining_qty, reduce_only=True
            )
            self.positions.remove(pos)
            self.logger.warning(f"Emergency closed {pos.side} position: {pos.trade_id}")

    def stop(self):
        """Stop the bot gracefully."""
        self.running = False
        self.logger.info("Bot stop requested.")


# ─────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Gold Momentum Multi-TF Scalper Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest on historical data
  python main.py --mode backtest --data data/XAUUSDT_5m.csv

  # Download historical data first
  python main.py --mode download --since 2024-01-01T00:00:00Z --output data/XAUUSDT_5m.csv

  # Paper trading (simulated)
  python main.py --mode paper

  # Live trading (uses config.json testnet/live setting)
  python main.py --mode live

  # Specify custom config
  python main.py --mode backtest --config my_config.json --data data/XAUUSDT_5m.csv
        """
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['backtest', 'paper', 'live', 'download'],
        help='Bot mode: backtest, paper, live, or download'
    )
    parser.add_argument(
        '--config', type=str, default='config.json',
        help='Path to config JSON file (default: config.json)'
    )
    parser.add_argument(
        '--data', type=str, default=None,
        help='Path to historical data file (CSV/Parquet) for backtest mode'
    )
    parser.add_argument(
        '--data-1h', type=str, default=None,
        help='Optional: separate 1H data file. If not provided, 5M data is resampled.'
    )
    parser.add_argument(
        '--since', type=str, default='2024-01-01T00:00:00Z',
        help='Start date for data download (ISO format)'
    )
    parser.add_argument(
        '--output', type=str, default='data/XAUUSDT_5m.csv',
        help='Output file path for downloaded data'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip equity curve plot generation in backtest mode'
    )
    return parser.parse_args()


def run_backtest(args, config: ConfigLoader, logger: logging.Logger):
    """Execute backtest mode."""
    if not args.data:
        logger.error("Backtest mode requires --data flag with path to historical data file.")
        logger.info("Example: python main.py --mode backtest --data data/XAUUSDT_5m.csv")
        logger.info("To download data first: python main.py --mode download --since 2024-01-01T00:00:00Z")
        sys.exit(1)

    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    # Load data
    logger.info(f"Loading 5M data from: {args.data}")
    df_5m = DataLoader.load(args.data, '5m')
    logger.info(f"Loaded {len(df_5m)} bars — {df_5m.iloc[0]['timestamp']} to {df_5m.iloc[-1]['timestamp']}")

    df_1h = None
    if args.data_1h:
        logger.info(f"Loading 1H data from: {args.data_1h}")
        df_1h = DataLoader.load(args.data_1h, '1h')

    # Run backtester
    backtester = Backtester(config, logger)
    result = backtester.run(df_5m, df_1h)

    # Plot
    if not args.no_plot:
        backtester.plot_equity_curve(result)

    # Export trades
    backtester.export_trades_csv(result)

    return result


def run_download(args, config: ConfigLoader, logger: logging.Logger):
    """Download historical data from exchange."""
    symbol = config['symbol']
    exchange_name = config.get('exchange', 'name', default='binance')
    testnet = config.get('exchange', 'testnet', default=True)

    logger.info(f"Downloading {symbol} 5M data from {exchange_name} since {args.since}...")

    df = DataLoader.download_from_exchange(
        exchange_name=exchange_name,
        symbol=symbol,
        timeframe='5m',
        since=args.since,
        testnet=testnet
    )

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    df.to_csv(args.output, index=False)
    logger.info(f"✅ Downloaded {len(df)} candles → {args.output}")

    # Also download 1H
    output_1h = args.output.replace('5m', '1h')
    logger.info(f"Downloading 1H data...")
    df_1h = DataLoader.download_from_exchange(
        exchange_name=exchange_name,
        symbol=symbol,
        timeframe='1h',
        since=args.since,
        testnet=testnet
    )
    df_1h.to_csv(output_1h, index=False)
    logger.info(f"✅ Downloaded {len(df_1h)} 1H candles → {output_1h}")


def run_live_or_paper(args, config: ConfigLoader, mode: str):
    """Run live or paper trading."""
    bot = TradingBot(config, mode=mode)

    # Graceful shutdown
    def signal_handler(sig, frame):
        bot.logger.info("\n⏹ Shutdown signal received...")
        bot.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run async event loop
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        bot.logger.info("Bot stopped.")


def main():
    args = parse_args()

    # Load config
    config = ConfigLoader(args.config)
    logger = setup_logging(config)

    logger.info("=" * 60)
    logger.info("  GOLD MOMENTUM MULTI-TF SCALPER BOT")
    logger.info(f"  Symbol: {config['symbol']}")
    logger.info(f"  Mode:   {args.mode.upper()}")
    logger.info("=" * 60)

    if args.mode == 'backtest':
        run_backtest(args, config, logger)

    elif args.mode == 'download':
        run_download(args, config, logger)

    elif args.mode in ('paper', 'live'):
        run_live_or_paper(args, config, args.mode)

    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
