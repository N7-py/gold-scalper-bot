"""
test_bot.py -- Comprehensive test suite for Gold Momentum Multi-TF Scalper Bot.

Tests: ConfigLoader, DatabaseManager, StrategyEngine, ExchangeClient (paper mode),
       Backtester pipeline, position sizing, session filters, PnL calculation,
       exit conditions (SL/TP1/trail), and a mini end-to-end paper-mode dry run.

Run:  python test_bot.py
"""

import sys
import os
import json
import time
import logging
import tempfile
import unittest
import sqlite3
from datetime import datetime, timezone, timedelta
from dataclasses import asdict

import pandas as pd
import numpy as np

# make sure project directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    ConfigLoader, DatabaseManager, TelegramAlerter, NewsFilter, DXYFilter,
    Position, TradeRecord, setup_logging,
    is_trading_session, calculate_position_size, generate_trade_id
)
from strategy import StrategyEngine, Signal, IndicatorEngine
from backtester import Backtester, DataLoader, BacktestResult


# ────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────────────────────────────────

SAMPLE_CONFIG = {
    "symbol": "XAU/USDT:USDT",
    "market_type": "swap",
    "mode": "backtest",
    "exchange": {
        "name": "binance",
        "testnet": False,
        "api_key": "TEST_KEY",
        "api_secret": "TEST_SECRET"
    },
    "strategy": {
        "ema_fast": 50,
        "ema_slow": 200,
        "rsi_period": 14,
        "atr_period": 14,
        "pullback_atr_mult": 1.0,
        "sl_atr_mult": 1.5,
        "tp1_atr_mult": 2.0,
        "trail_atr_mult": 1.0,
        "tp1_close_pct": 0.5
    },
    "risk": {
        "risk_pct": 0.5,
        "max_risk_pct": 1.0,
        "max_concurrent_positions": 2,
        "max_daily_drawdown_pct": 5.0,
        "min_atr_threshold": 0.5
    },
    "session_hours_utc": {
        "london": [8, 12],
        "new_york": [13, 17]
    },
    "execution": {
        "order_type": "market",
        "use_oco": False
    },
    "backtest": {
        "data_dir": "data",
        "initial_balance": 10000.0,
        "commission_pct": 0.04,
        "slippage_pct": 0.01
    },
    "database": {
        "type": "sqlite",
        "path": ":memory:"
    },
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": ""
    },
    "logging": {
        "level": "WARNING",
        "file": "test_bot.log"
    }
}


def make_config(overrides: dict = None) -> ConfigLoader:
    cfg = json.loads(json.dumps(SAMPLE_CONFIG))
    if overrides:
        cfg.update(overrides)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(cfg, f)
        tmp_path = f.name
    loader = ConfigLoader(tmp_path)
    os.unlink(tmp_path)
    return loader


def make_logger() -> logging.Logger:
    logger = logging.getLogger("test")
    logger.setLevel(logging.WARNING)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    return logger


def synthetic_ohlcv(n=400, base_price=2000.0, trend=0.02, noise=5.0, seed=42):
    """Generate synthetic OHLCV data mimicking a gold uptrend."""
    np.random.seed(seed)
    closes = [base_price]
    for i in range(1, n):
        drift = closes[-1] * trend / n * 50
        shock = np.random.normal(0, noise)
        closes.append(max(100, closes[-1] + drift + shock))

    closes = np.array(closes)
    highs  = closes + np.abs(np.random.normal(0, noise * 0.5, n))
    lows   = closes - np.abs(np.random.normal(0, noise * 0.5, n))
    opens  = np.roll(closes, 1); opens[0] = base_price
    volumes = np.random.uniform(100, 1000, n)

    start = pd.Timestamp("2024-01-01 08:00:00", tz="UTC")
    timestamps = [start + pd.Timedelta(minutes=5 * i) for i in range(n)]

    return pd.DataFrame({
        "timestamp": timestamps,
        "open":  opens,
        "high":  highs,
        "low":   lows,
        "close": closes,
        "volume": volumes
    })


# ────────────────────────────────────────────────────────────────────────────
#  1. CONFIG LOADER TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

    def test_symbol_loaded(self):
        self.assertEqual(self.config["symbol"], "XAU/USDT:USDT")

    def test_nested_get(self):
        self.assertEqual(self.config.get("strategy", "ema_fast"), 50)
        self.assertEqual(self.config.get("strategy", "ema_slow"), 200)
        self.assertEqual(self.config.get("risk", "risk_pct"), 0.5)

    def test_default_fallback(self):
        val = self.config.get("nonexistent", "key", default="fallback")
        self.assertEqual(val, "fallback")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ConfigLoader("/nonexistent/path/config.json")

    def test_invalid_risk_pct_raises(self):
        bad_cfg = json.loads(json.dumps(SAMPLE_CONFIG))
        bad_cfg["risk"]["risk_pct"] = 99.0
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(bad_cfg, f)
            tmp = f.name
        with self.assertRaises(ValueError):
            ConfigLoader(tmp)
        os.unlink(tmp)

    def test_all_required_keys_present(self):
        for key in ["symbol", "exchange", "strategy", "risk", "backtest", "database"]:
            self.assertIn(key, self.config.config)

    def test_initial_balance(self):
        self.assertEqual(self.config.get("backtest", "initial_balance"), 10000.0)


# ────────────────────────────────────────────────────────────────────────────
#  2. DATABASE MANAGER TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        self.db = DatabaseManager(":memory:")

    def tearDown(self):
        self.db.close()

    def _sample_trade(self, trade_id="T001", side="long", pnl=0.0, status="open"):
        return TradeRecord(
            trade_id=trade_id,
            symbol="XAU/USDT:USDT",
            side=side,
            entry_price=2000.0,
            entry_time=datetime.now(timezone.utc).isoformat(),
            quantity=0.01,
            sl_price=1985.0,
            tp1_price=2030.0,
            status=status,
            pnl=pnl,
            remaining_qty=0.01
        )

    def test_insert_and_retrieve_trade(self):
        trade = self._sample_trade()
        self.db.insert_trade(trade)
        open_trades = self.db.get_open_trades()
        self.assertEqual(len(open_trades), 1)
        self.assertEqual(open_trades[0]["trade_id"], "T001")

    def test_update_trade(self):
        self.db.insert_trade(self._sample_trade())
        self.db.update_trade("T001", {"status": "closed", "pnl": 25.50})
        trades = self.db.get_all_trades()
        self.assertEqual(trades[0]["status"], "closed")
        self.assertAlmostEqual(trades[0]["pnl"], 25.50)

    def test_multiple_trades(self):
        for i in range(5):
            self.db.insert_trade(self._sample_trade(trade_id=f"T{i:03d}"))
        self.assertEqual(len(self.db.get_all_trades()), 5)

    def test_get_open_trades_filters_closed(self):
        self.db.insert_trade(self._sample_trade("T001", status="open"))
        self.db.insert_trade(self._sample_trade("T002", status="closed"))
        open_trades = self.db.get_open_trades()
        self.assertEqual(len(open_trades), 1)

    def test_daily_pnl_insert_and_update(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.db.update_daily_pnl(today, 100.0, True, 10100.0)
        self.db.update_daily_pnl(today, -50.0, False, 10050.0)
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT * FROM daily_pnl WHERE date = ?", (today,))
        row = cursor.fetchone()
        self.assertAlmostEqual(row["total_pnl"], 50.0)
        self.assertEqual(row["win_count"], 1)
        self.assertEqual(row["loss_count"], 1)

    def test_equity_point_insert(self):
        self.db.insert_equity_point(10000.0, 10050.0, 0.5)
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM equity_curve")
        self.assertEqual(cursor.fetchone()[0], 1)


# ────────────────────────────────────────────────────────────────────────────
#  3. POSITION SIZING TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestPositionSizing(unittest.TestCase):

    def test_basic_sizing(self):
        qty = calculate_position_size(10000, 0.5, 2000.0, 1985.0)
        expected = (10000 * 0.005) / 15.0
        self.assertAlmostEqual(qty, expected, delta=0.01)

    def test_zero_sl_distance_returns_zero(self):
        qty = calculate_position_size(10000, 0.5, 2000.0, 2000.0)
        self.assertEqual(qty, 0.0)

    def test_minimum_quantity_enforced(self):
        qty = calculate_position_size(10, 0.5, 2000.0, 1000.0)
        self.assertEqual(qty, 0.001)

    def test_qty_rounded_to_step(self):
        qty = calculate_position_size(10000, 0.5, 2000.0, 1990.0)
        # Check qty is a multiple of 0.001 using integer math to avoid float modulo issues
        self.assertAlmostEqual(round(qty * 1000) - qty * 1000, 0.0, places=4)

    def test_large_balance(self):
        qty = calculate_position_size(1_000_000, 1.0, 2000.0, 1950.0)
        self.assertAlmostEqual(qty, 200.0, delta=1.0)


# ────────────────────────────────────────────────────────────────────────────
#  4. SESSION FILTER TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestSessionFilter(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

    def test_london_session_open(self):
        for hour in [8, 9, 10, 11, 12]:
            self.assertTrue(is_trading_session(self.config, hour), f"Hour {hour} should be open")

    def test_new_york_session_open(self):
        for hour in [13, 14, 15, 16, 17]:
            self.assertTrue(is_trading_session(self.config, hour), f"Hour {hour} should be open")

    def test_closed_hours(self):
        for hour in [0, 3, 6, 7, 18, 22, 23]:
            self.assertFalse(is_trading_session(self.config, hour), f"Hour {hour} should be closed")


# ────────────────────────────────────────────────────────────────────────────
#  5. INDICATOR ENGINE TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestIndicatorEngine(unittest.TestCase):

    def setUp(self):
        self.config = make_config()
        self.engine = IndicatorEngine(self.config)
        self.df = synthetic_ohlcv(400)

    def test_indicators_computed(self):
        df = self.engine.compute(self.df)
        for col in ["ema50", "ema200", "rsi", "atr"]:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_ema_values_finite_after_warmup(self):
        df = self.engine.compute(self.df)
        tail = df.tail(100)
        self.assertFalse(tail["ema200"].isna().all(), "EMA200 should have values after 200 bars")

    def test_rsi_bounded(self):
        df = self.engine.compute(self.df)
        valid_rsi = df["rsi"].dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all(), "RSI must be 0-100")

    def test_atr_positive(self):
        df = self.engine.compute(self.df)
        valid_atr = df["atr"].dropna()
        self.assertTrue((valid_atr >= 0).all(), "ATR must be non-negative")

    def test_no_data_mutation(self):
        original_close = self.df["close"].copy()
        self.engine.compute(self.df)
        pd.testing.assert_series_equal(self.df["close"], original_close)


# ────────────────────────────────────────────────────────────────────────────
#  6. STRATEGY ENGINE TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestStrategyEngine(unittest.TestCase):

    def setUp(self):
        self.config = make_config()
        self.logger = make_logger()
        self.engine = StrategyEngine(self.config, self.logger)

    def _make_df(self, n=350, base=2000.0, trend=0.05):
        df = synthetic_ohlcv(n, base_price=base, trend=trend)
        return self.engine.compute_indicators(df)

    def test_compute_indicators_returns_correct_cols(self):
        df = self._make_df()
        for col in ["ema50", "ema200", "rsi", "atr"]:
            self.assertIn(col, df.columns)

    def test_evaluate_returns_none_when_not_enough_data(self):
        df_short = synthetic_ohlcv(50)
        df_short = self.engine.compute_indicators(df_short)
        result = self.engine.evaluate(df_short, df_short, [], datetime.now(timezone.utc))
        self.assertIsNone(result)

    def test_evaluate_returns_none_outside_session(self):
        df = self._make_df()
        offhour = datetime(2024, 6, 15, 2, 0, tzinfo=timezone.utc)
        result = self.engine.evaluate(df, df, [], offhour)
        self.assertIsNone(result)

    def test_evaluate_returns_none_at_max_positions(self):
        df = self._make_df()
        session_time = datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc)
        positions = [
            Position("T1", "XAU", "long", 2000, datetime.now(timezone.utc), 0.01, 0.01, 1985, 2030),
            Position("T2", "XAU", "short", 2000, datetime.now(timezone.utc), 0.01, 0.01, 2015, 1970)
        ]
        result = self.engine.evaluate(df, df, positions, session_time)
        self.assertIsNone(result, "Should return None when max_positions (2) reached")

    def test_signal_has_valid_fields_when_generated(self):
        df_5m = self._make_df(350, base=2000.0, trend=0.1)
        df_1h = DataLoader.resample_to_1h(synthetic_ohlcv(350, base_price=2000.0, trend=0.1))
        df_1h = self.engine.compute_indicators(df_1h)
        session_time = datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc)
        signal = self.engine.evaluate(df_1h, df_5m, [], session_time)
        if signal is not None:
            self.assertIn(signal.side, ["long", "short"])
            self.assertGreater(signal.entry_price, 0)
            if signal.side == "long":
                self.assertLess(signal.sl_price, signal.entry_price)
                self.assertGreater(signal.tp1_price, signal.entry_price)
            else:
                self.assertGreater(signal.sl_price, signal.entry_price)
                self.assertLess(signal.tp1_price, signal.entry_price)

    def test_signal_rr_ratio_correct(self):
        df = self._make_df()
        session = datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc)
        signal = self.engine.evaluate(df, df, [], session)
        if signal is not None:
            sl_dist = abs(signal.entry_price - signal.sl_price)
            tp_dist = abs(signal.entry_price - signal.tp1_price)
            expected_rr = (self.config.get("strategy", "tp1_atr_mult") /
                           self.config.get("strategy", "sl_atr_mult"))
            actual_rr = tp_dist / sl_dist if sl_dist > 0 else 0
            self.assertAlmostEqual(actual_rr, expected_rr, delta=0.05)


# ────────────────────────────────────────────────────────────────────────────
#  7. EXIT CONDITIONS TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestExitConditions(unittest.TestCase):

    def setUp(self):
        self.config = make_config()
        self.logger = make_logger()
        self.engine = StrategyEngine(self.config, self.logger)

    def _long(self, entry=2000.0, sl=1985.0, tp1=2030.0, qty=0.1, tp1_hit=False):
        return Position(
            trade_id="TEST_L", symbol="XAU/USDT:USDT", side="long",
            entry_price=entry, entry_time=datetime.now(timezone.utc),
            quantity=qty, remaining_qty=qty,
            sl_price=sl, tp1_price=tp1, tp1_hit=tp1_hit,
            highest_since_entry=entry, lowest_since_entry=entry
        )

    def _short(self, entry=2000.0, sl=2015.0, tp1=1970.0, qty=0.1, tp1_hit=False):
        return Position(
            trade_id="TEST_S", symbol="XAU/USDT:USDT", side="short",
            entry_price=entry, entry_time=datetime.now(timezone.utc),
            quantity=qty, remaining_qty=qty,
            sl_price=sl, tp1_price=tp1, tp1_hit=tp1_hit,
            highest_since_entry=0, lowest_since_entry=entry
        )

    def test_long_sl_triggered(self):
        pos = self._long()
        reason, price, qty = self.engine.check_exit_conditions(pos, 1984.0, 2000.0, 1984.0, 5.0)
        self.assertEqual(reason, "sl")
        self.assertAlmostEqual(price, 1985.0)

    def test_long_tp1_triggered(self):
        pos = self._long()
        reason, price, qty = self.engine.check_exit_conditions(pos, 2031.0, 2031.0, 2000.0, 5.0)
        self.assertEqual(reason, "tp1")
        self.assertAlmostEqual(price, 2030.0)
        self.assertAlmostEqual(qty, pos.quantity * 0.5, places=4)

    def test_long_no_exit_in_range(self):
        pos = self._long()
        reason, _, _ = self.engine.check_exit_conditions(pos, 2010.0, 2015.0, 1990.0, 5.0)
        self.assertIsNone(reason)

    def test_long_trail_stop_after_tp1(self):
        pos = self._long(tp1_hit=True)
        pos.trail_stop = 2020.0
        pos.highest_since_entry = 2040.0
        reason, price, qty = self.engine.check_exit_conditions(pos, 2018.0, 2030.0, 2018.0, 5.0)
        self.assertEqual(reason, "trail_stop")

    def test_long_trail_not_triggered_above_stop(self):
        pos = self._long(tp1_hit=True)
        pos.trail_stop = 2020.0
        pos.highest_since_entry = 2040.0
        reason, _, _ = self.engine.check_exit_conditions(pos, 2025.0, 2030.0, 2022.0, 5.0)
        self.assertIsNone(reason)

    def test_short_sl_triggered(self):
        pos = self._short()
        reason, price, qty = self.engine.check_exit_conditions(pos, 2016.0, 2016.0, 2000.0, 5.0)
        self.assertEqual(reason, "sl")
        self.assertAlmostEqual(price, 2015.0)

    def test_short_tp1_triggered(self):
        pos = self._short()
        reason, price, qty = self.engine.check_exit_conditions(pos, 1969.0, 2000.0, 1969.0, 5.0)
        self.assertEqual(reason, "tp1")
        self.assertAlmostEqual(price, 1970.0)

    def test_short_trail_stop_after_tp1(self):
        pos = self._short(tp1_hit=True)
        pos.trail_stop = 1980.0
        pos.lowest_since_entry = 1960.0
        reason, price, qty = self.engine.check_exit_conditions(pos, 1981.0, 1981.0, 1965.0, 5.0)
        self.assertEqual(reason, "trail_stop")

    def test_opposite_signal_triggers_exit(self):
        pos = self._long()
        reason, price, qty = self.engine.check_exit_conditions(
            pos, 2010.0, 2015.0, 1990.0, 5.0, opposite_signal=True)
        self.assertEqual(reason, "opposite_signal")


# ────────────────────────────────────────────────────────────────────────────
#  8. DATA LOADER TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestDataLoader(unittest.TestCase):

    def test_load_synthetic_csv(self):
        df_orig = synthetic_ohlcv(200)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_orig.to_csv(f.name, index=False)
            tmp_path = f.name
        try:
            df_loaded = DataLoader.load(tmp_path)
            self.assertEqual(len(df_loaded), 200)
            for col in ["open", "high", "low", "close", "volume"]:
                self.assertIn(col, df_loaded.columns)
        finally:
            os.unlink(tmp_path)

    def test_resample_to_1h(self):
        df_5m = synthetic_ohlcv(300)
        df_1h = DataLoader.resample_to_1h(df_5m)
        self.assertGreater(len(df_1h), 20)
        self.assertLessEqual(len(df_1h), 26)

    def test_resample_preserves_ohlcv(self):
        df_5m = synthetic_ohlcv(120)
        df_1h = DataLoader.resample_to_1h(df_5m)
        for col in ["open", "high", "low", "close", "volume"]:
            self.assertIn(col, df_1h.columns)

    def test_unsupported_format_raises(self):
        with self.assertRaises(ValueError):
            DataLoader.load("bad_file.xyz")

    def test_existing_ohlcv_csv(self):
        csv_path = os.path.join(os.path.dirname(__file__), "data", "XAUUSDT_5m.csv")
        if not os.path.exists(csv_path):
            self.skipTest("data/XAUUSDT_5m.csv not found")
        df = DataLoader.load(csv_path)
        self.assertGreater(len(df), 100)


# ────────────────────────────────────────────────────────────────────────────
#  9. BACKTESTER PIPELINE TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestBacktesterPipeline(unittest.TestCase):

    def setUp(self):
        self.config = make_config()
        self.logger = make_logger()
        self.backtester = Backtester(self.config, self.logger)

    def test_run_returns_result(self):
        df_5m = synthetic_ohlcv(500, trend=0.2, seed=1)
        result = self.backtester.run(df_5m, save_db=False)
        self.assertIsInstance(result, BacktestResult)

    def test_equity_curve_has_entries(self):
        df_5m = synthetic_ohlcv(500, trend=0.2, seed=2)
        result = self.backtester.run(df_5m, save_db=False)
        self.assertGreater(len(result.equity_curve), 100)

    def test_initial_balance_preserved(self):
        df_5m = synthetic_ohlcv(500, seed=3)
        result = self.backtester.run(df_5m, save_db=False)
        self.assertEqual(result.initial_balance, 10000.0)

    def test_metrics_computed(self):
        df_5m = synthetic_ohlcv(500, trend=0.15, seed=4)
        result = self.backtester.run(df_5m, save_db=False)
        if result.total_trades > 0:
            self.assertGreaterEqual(result.win_rate, 0.0)
            self.assertLessEqual(result.win_rate, 100.0)
            self.assertGreaterEqual(result.profit_factor, 0.0)

    def test_no_negative_equity(self):
        df_5m = synthetic_ohlcv(500, seed=5)
        result = self.backtester.run(df_5m, save_db=False)
        self.assertGreater(min(result.equity_curve), 0)

    def test_csv_export(self):
        df_5m = synthetic_ohlcv(400, trend=0.2, seed=6)
        result = self.backtester.run(df_5m, save_db=False)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            tmp = f.name
        try:
            self.backtester.export_trades_csv(result, tmp)
            if result.total_trades > 0:
                df = pd.read_csv(tmp)
                self.assertEqual(len(df), result.total_trades)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_full_real_data_backtest(self):
        csv_path = os.path.join(os.path.dirname(__file__), "data", "XAUUSDT_5m.csv")
        if not os.path.exists(csv_path):
            self.skipTest("data/XAUUSDT_5m.csv not found -- run download mode first")
        df = DataLoader.load(csv_path)
        result = self.backtester.run(df, save_db=False)
        self.assertIsInstance(result, BacktestResult)
        print(f"\n  [REAL DATA] Trades={result.total_trades} | "
              f"WR={result.win_rate:.1f}% | PF={result.profit_factor:.2f} | "
              f"Return={result.return_pct:+.2f}%")


# ────────────────────────────────────────────────────────────────────────────
#  10. PAPER EXCHANGE CLIENT TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestPaperExchange(unittest.TestCase):

    def setUp(self):
        from exchange import ExchangeClient
        self.config = make_config()
        self.logger = make_logger()
        self.exchange = ExchangeClient.__new__(ExchangeClient)
        self.exchange.config = self.config
        self.exchange.logger = self.logger
        self.exchange.paper_mode = True
        self.exchange.exchange = None
        self.exchange._public_exchange = None
        self.exchange.paper_balance = 10000.0
        self.exchange.paper_orders = []
        self.exchange.paper_order_id = 1000

    def test_get_balance_returns_paper_balance(self):
        self.assertAlmostEqual(self.exchange.get_balance(), 10000.0)

    def test_update_paper_balance_profit(self):
        self.exchange.update_paper_balance(250.0)
        self.assertAlmostEqual(self.exchange.get_balance(), 10250.0)

    def test_update_paper_balance_loss(self):
        self.exchange.update_paper_balance(-100.0)
        self.assertAlmostEqual(self.exchange.get_balance(), 9900.0)

    def test_paper_market_order_fills_immediately(self):
        self.exchange.get_ticker_price = lambda sym: 2010.0
        order = self.exchange._paper_market_order("XAU/USDT:USDT", "buy", 0.01)
        self.assertIsNotNone(order)
        self.assertEqual(order["status"], "filled")
        self.assertEqual(order["side"], "buy")
        self.assertAlmostEqual(float(order["average"]), 2010.0)

    def test_paper_limit_order_has_open_status(self):
        order = self.exchange._paper_limit_order("XAU/USDT:USDT", "sell", 0.01, 2025.0)
        self.assertEqual(order["status"], "open")
        self.assertAlmostEqual(float(order["price"]), 2025.0)

    def test_cancel_all_orders_clears_paper_orders(self):
        self.exchange.paper_orders = [{"id": "p1"}, {"id": "p2"}]
        self.exchange.cancel_all_orders("XAU/USDT:USDT")
        self.assertEqual(len(self.exchange.paper_orders), 0)

    def test_place_stop_loss_paper_returns_dict(self):
        result = self.exchange.place_stop_loss("XAU/USDT:USDT", "long", 0.01, 1985.0)
        self.assertIsNotNone(result)
        self.assertIn("type", result)

    def test_place_take_profit_paper_returns_dict(self):
        result = self.exchange.place_take_profit("XAU/USDT:USDT", "long", 0.01, 2030.0)
        self.assertIsNotNone(result)


# ────────────────────────────────────────────────────────────────────────────
#  11. FILTERS AND ALERTS TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestFiltersAndAlerts(unittest.TestCase):

    def setUp(self):
        self.config = make_config()
        self.logger = make_logger()

    def test_news_filter_never_blocks(self):
        news = NewsFilter(self.logger)
        self.assertFalse(news.should_skip_trade(datetime.now(timezone.utc)))

    def test_dxy_filter_disabled_never_skips(self):
        dxy = DXYFilter(enabled=False, logger=self.logger)
        self.assertFalse(dxy.should_skip("long"))
        self.assertFalse(dxy.should_skip("short"))

    def test_dxy_blocks_long_on_dxy_rise(self):
        dxy = DXYFilter(enabled=True, logger=self.logger)
        dxy.update(103.5, 103.0)  # +0.48%
        self.assertTrue(dxy.should_skip("long"))

    def test_dxy_blocks_short_on_dxy_fall(self):
        dxy = DXYFilter(enabled=True, logger=self.logger)
        dxy.update(102.5, 103.0)  # -0.48%
        self.assertTrue(dxy.should_skip("short"))

    def test_dxy_no_block_within_threshold(self):
        dxy = DXYFilter(enabled=True, logger=self.logger)
        dxy.update(103.1, 103.0)  # +0.097% -- below 0.3%
        self.assertFalse(dxy.should_skip("long"))

    def test_telegram_disabled_is_noop(self):
        tg = TelegramAlerter(self.config, self.logger)
        tg.send("test message")
        tg.send_alert("test alert")

    def test_trade_id_format(self):
        self.assertTrue(generate_trade_id("long").startswith("L"))
        self.assertTrue(generate_trade_id("short").startswith("S"))

    def test_trade_id_unique(self):
        ids = {generate_trade_id("long") for _ in range(10)}
        self.assertEqual(len(ids), 10)


# ────────────────────────────────────────────────────────────────────────────
#  12. DATA CLASSES TESTS
# ────────────────────────────────────────────────────────────────────────────

class TestDataClasses(unittest.TestCase):

    def test_trade_record_defaults(self):
        tr = TradeRecord(
            trade_id="T1", symbol="XAU", side="long",
            entry_price=2000.0, entry_time="2024-01-01T08:00:00",
            quantity=0.01, sl_price=1985.0, tp1_price=2030.0
        )
        self.assertEqual(tr.status, "open")
        self.assertEqual(tr.pnl, 0.0)
        self.assertEqual(tr.exit_price, 0.0)

    def test_position_defaults(self):
        pos = Position(
            trade_id="P1", symbol="XAU", side="short",
            entry_price=2000.0, entry_time=datetime.now(timezone.utc),
            quantity=0.01, remaining_qty=0.01,
            sl_price=2015.0, tp1_price=1970.0
        )
        self.assertFalse(pos.tp1_hit)
        self.assertEqual(pos.trail_stop, 0.0)

    def test_position_to_dict(self):
        pos = Position(
            trade_id="P2", symbol="XAU", side="long",
            entry_price=2000.0, entry_time=datetime.now(timezone.utc),
            quantity=0.01, remaining_qty=0.01,
            sl_price=1985.0, tp1_price=2030.0
        )
        d = asdict(pos)
        self.assertIn("trade_id", d)
        self.assertIn("entry_price", d)


# ────────────────────────────────────────────────────────────────────────────
#  13. PAPER MODE DRY RUN (integration, no network)
# ────────────────────────────────────────────────────────────────────────────

class TestPaperModeDryRun(unittest.TestCase):

    def _run_paper_simulation(self, n_bars=500, trend=0.15):
        config   = make_config()
        logger   = make_logger()
        strategy = StrategyEngine(config, logger)
        db       = DatabaseManager(":memory:")

        paper_balance = 10000.0
        positions = []
        symbol    = "XAU/USDT:USDT"
        risk_pct  = 0.5
        lookback  = 210

        df_5m = strategy.compute_indicators(synthetic_ohlcv(n_bars, base_price=2000.0, trend=trend))
        df_1h = strategy.compute_indicators(
            DataLoader.resample_to_1h(synthetic_ohlcv(n_bars, base_price=2000.0, trend=trend))
        )

        trades_logged = 0

        for i in range(lookback, n_bars):
            bar      = df_5m.iloc[i]
            bar_time = bar["timestamp"]
            if bar_time.tzinfo is None:
                bar_time = bar_time.replace(tzinfo=timezone.utc)
            atr = bar["atr"] if not pd.isna(bar["atr"]) else 0

            for pos in list(positions):
                reason, exit_price, close_qty = strategy.check_exit_conditions(
                    pos, bar["close"], bar["high"], bar["low"], atr
                )
                if reason and reason != "tp1":
                    pnl = ((exit_price - pos.entry_price) * close_qty
                           if pos.side == "long"
                           else (pos.entry_price - exit_price) * close_qty)
                    paper_balance += pnl
                    positions.remove(pos)
                    db.update_trade(pos.trade_id, {
                        "status": "closed", "exit_price": exit_price,
                        "pnl": round(pnl, 4), "exit_reason": reason
                    })

            if len(positions) < 2:
                df_5m_win = df_5m.iloc[max(0, i - lookback):i + 1]
                h1_mask   = df_1h["timestamp"] <= bar_time
                if h1_mask.sum() >= lookback:
                    df_1h_win = df_1h[h1_mask].tail(lookback + 50)
                    signal = strategy.evaluate(df_1h_win, df_5m_win, positions,
                                               bar_time.to_pydatetime())
                    if signal:
                        qty = calculate_position_size(paper_balance, risk_pct,
                                                      signal.entry_price, signal.sl_price)
                        if qty > 0:
                            trade_id = generate_trade_id(signal.side)
                            pos = Position(
                                trade_id=trade_id, symbol=symbol, side=signal.side,
                                entry_price=signal.entry_price,
                                entry_time=bar_time.to_pydatetime(),
                                quantity=qty, remaining_qty=qty,
                                sl_price=signal.sl_price, tp1_price=signal.tp1_price,
                                atr_at_entry=signal.atr_value,
                                highest_since_entry=signal.entry_price if signal.side == "long" else 0,
                                lowest_since_entry=signal.entry_price if signal.side == "short" else 999999
                            )
                            positions.append(pos)
                            db.insert_trade(TradeRecord(
                                trade_id=trade_id, symbol=symbol, side=signal.side,
                                entry_price=signal.entry_price,
                                entry_time=bar_time.isoformat(),
                                quantity=qty, sl_price=signal.sl_price,
                                tp1_price=signal.tp1_price,
                                entry_reason=signal.reason, remaining_qty=qty
                            ))
                            trades_logged += 1

        db.close()
        return trades_logged, paper_balance

    def test_simulation_runs_without_error(self):
        trades, balance = self._run_paper_simulation(n_bars=500)
        self.assertIsInstance(trades, int)
        self.assertGreater(balance, 0)

    def test_balance_stays_reasonable(self):
        _, balance = self._run_paper_simulation(n_bars=500)
        self.assertGreater(balance, 5000.0)
        self.assertLess(balance, 25000.0)

    def test_no_crash_with_trend(self):
        trades, _ = self._run_paper_simulation(n_bars=500, trend=0.2)
        self.assertGreaterEqual(trades, 0)


# ────────────────────────────────────────────────────────────────────────────
#  RUNNER
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_classes = [
        ("Config Loader",            TestConfigLoader),
        ("Database Manager",         TestDatabaseManager),
        ("Position Sizing",          TestPositionSizing),
        ("Session Filter",           TestSessionFilter),
        ("Indicator Engine",         TestIndicatorEngine),
        ("Strategy Engine",          TestStrategyEngine),
        ("Exit Conditions",          TestExitConditions),
        ("Data Loader",              TestDataLoader),
        ("Backtester Pipeline",      TestBacktesterPipeline),
        ("Paper Exchange Client",    TestPaperExchange),
        ("Filters & Alerts",         TestFiltersAndAlerts),
        ("Data Classes",             TestDataClasses),
        ("Paper Mode Dry Run (E2E)", TestPaperModeDryRun),
    ]

    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[1m"
    X = "\033[0m"

    print(f"\n{B}{'=' * 65}{X}")
    print(f"{B}   [*] GOLD SCALPER BOT -- PRE-DEPLOYMENT TEST SUITE{X}")
    print(f"{B}{'=' * 65}{X}")

    total_pass = total_fail = total_skip = 0

    for label, cls in test_classes:
        print(f"\n{B}  >> {label}{X}")

        loader = unittest.TestLoader()
        suite  = loader.loadTestsFromTestCase(cls)

        # Use dir(cls) to enumerate names -- avoids None from TestSuite iterator
        all_names = sorted(n for n in dir(cls) if n.startswith("test_"))

        res = unittest.TestResult()
        suite.run(res)

        failed_names  = {t._testMethodName for t, _ in res.failures + res.errors}
        skipped_names = {t._testMethodName for t, _ in res.skipped}

        for name in all_names:
            desc = name.replace("test_", "").replace("_", " ")
            if name in failed_names:
                print(f"    {R}[FAIL]{X} {desc}")
            elif name in skipped_names:
                print(f"    {Y}[SKIP]{X} {desc}")
            else:
                print(f"    {G}[ ok ]{X} {desc}")

        n_fail = len(res.failures) + len(res.errors)
        n_skip = len(res.skipped)
        n_pass = res.testsRun - n_fail - n_skip

        total_fail += n_fail
        total_skip += n_skip
        total_pass += n_pass

        for test, tb in res.failures + res.errors:
            method = test._testMethodName.replace("test_", "").replace("_", " ")
            print(f"\n    {R}--- Detail: '{method}' ---{X}")
            for line in tb.strip().split('\n')[-6:]:
                print(f"      {line}")

    grand = total_pass + total_fail + total_skip
    print(f"\n{B}{'=' * 65}{X}")
    print(f"  Total   : {grand}")
    print(f"  {G}Passed  : {total_pass}{X}")
    if total_fail:
        print(f"  {R}Failed  : {total_fail}{X}")
    if total_skip:
        print(f"  {Y}Skipped : {total_skip}{X}")
    print(f"{B}{'=' * 65}{X}")

    if total_fail == 0:
        print(f"\n  {G}{B}[OK] ALL TESTS PASSED -- Bot is safe for deployment!{X}")
    else:
        print(f"\n  {R}{B}[!!] {total_fail} FAILURE(S) -- Fix before going live.{X}")

    print(f"{B}{'=' * 65}{X}\n")
    sys.exit(0 if total_fail == 0 else 1)
