"""
strategy.py — Gold Momentum Multi-TF Scalper Strategy Engine.

Implements the exact strategy rules:
  - 1H bias: price > EMA200 AND EMA50 > EMA200 for longs (opposite for shorts)
  - 5M entry: pullback to EMA50, RSI > 50, bullish candle, session window
  - SL: 1.5 * ATR(14), TP1: 2 * ATR(14), trailing remainder with 1 * ATR steps
  - Max 2 concurrent positions, 0.5% risk per trade (configurable)
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging

from utils import (
    ConfigLoader, Position, TradeRecord,
    is_trading_session, calculate_position_size, generate_trade_id
)


# ─────────────────────────────────────────────
#  INDICATOR CALCULATOR
# ─────────────────────────────────────────────

class IndicatorEngine:
    """Calculate all required indicators on DataFrames."""

    def __init__(self, config: ConfigLoader):
        self.ema_fast = config.get('strategy', 'ema_fast', default=50)
        self.ema_slow = config.get('strategy', 'ema_slow', default=200)
        self.rsi_period = config.get('strategy', 'rsi_period', default=14)
        self.atr_period = config.get('strategy', 'atr_period', default=14)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute EMA50, EMA200, RSI(14), ATR(14) on the dataframe.
        Expects columns: open, high, low, close, volume.
        """
        df = df.copy()

        # EMAs
        df['ema50'] = ta.ema(df['close'], length=self.ema_fast)
        df['ema200'] = ta.ema(df['close'], length=self.ema_slow)

        # RSI
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)

        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)

        return df


# ─────────────────────────────────────────────
#  SIGNAL GENERATOR
# ─────────────────────────────────────────────

@dataclass
class Signal:
    """Represents a trade signal."""
    side: str              # 'long' or 'short'
    entry_price: float
    sl_price: float
    tp1_price: float
    atr_value: float
    reason: str
    timestamp: datetime


class StrategyEngine:
    """
    Core strategy logic — evaluates 1H bias + 5M entry conditions.
    
    LONG entry conditions (ALL must be true on 5M bar close):
      1. 1H: price > EMA200 AND EMA50 > EMA200 (uptrend bias)
      2. 5M: price pulled back to/near EMA50 from above
      3. 5M: RSI(14) > 50
      4. 5M: candle is bullish (close > open)
      5. Current hour is in London (8-12 UTC) or NY overlap (13-17 UTC)
    
    SHORT entry: exact opposite of all above.
    """

    def __init__(self, config: ConfigLoader, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.indicators = IndicatorEngine(config)

        # Strategy params
        self.pullback_atr_mult = config.get('strategy', 'pullback_atr_mult', default=1.0)
        self.sl_atr_mult = config.get('strategy', 'sl_atr_mult', default=1.5)
        self.tp1_atr_mult = config.get('strategy', 'tp1_atr_mult', default=2.0)
        self.trail_atr_mult = config.get('strategy', 'trail_atr_mult', default=1.0)
        self.tp1_close_pct = config.get('strategy', 'tp1_close_pct', default=0.5)
        self.min_atr = config.get('risk', 'min_atr_threshold', default=0.5)
        self.max_positions = config.get('risk', 'max_concurrent_positions', default=2)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Public wrapper for indicator computation."""
        return self.indicators.compute(df)

    def evaluate(
        self,
        df_1h: pd.DataFrame,
        df_5m: pd.DataFrame,
        current_positions: List[Position],
        utc_now: datetime
    ) -> Optional[Signal]:
        """
        Evaluate strategy conditions and return a Signal if entry criteria are met.
        Returns None if no valid signal.
        """
        # ── Pre-checks ──────────────────────────────
        # Max concurrent positions
        if len(current_positions) >= self.max_positions:
            return None

        # Need enough data
        if len(df_5m) < 210 or len(df_1h) < 210:
            return None

        # ── Get latest bars ──────────────────────────
        bar_5m = df_5m.iloc[-1]
        bar_5m_prev = df_5m.iloc[-2]
        bar_1h = df_1h.iloc[-1]

        # ── ATR filter ───────────────────────────────
        atr = bar_5m['atr']
        if pd.isna(atr) or atr < self.min_atr:
            return None

        # ── Session filter ───────────────────────────
        utc_hour = utc_now.hour
        if not is_trading_session(self.config, utc_hour):
            return None

        # ── 1H bias ──────────────────────────────────
        h1_close = bar_1h['close']
        h1_ema50 = bar_1h['ema50']
        h1_ema200 = bar_1h['ema200']

        if pd.isna(h1_ema50) or pd.isna(h1_ema200):
            return None

        uptrend_bias = (h1_close > h1_ema200) and (h1_ema50 > h1_ema200)
        downtrend_bias = (h1_close < h1_ema200) and (h1_ema50 < h1_ema200)

        # ── 5M conditions ───────────────────────────
        m5_close = bar_5m['close']
        m5_open = bar_5m['open']
        m5_ema50 = bar_5m['ema50']
        m5_ema200 = bar_5m['ema200']
        m5_rsi = bar_5m['rsi']
        m5_prev_close = bar_5m_prev['close']
        m5_prev_ema50 = bar_5m_prev.get('ema50', None)

        if pd.isna(m5_ema50) or pd.isna(m5_rsi) or pd.isna(atr):
            return None

        # Pullback threshold: price is within pullback_atr_mult * ATR of EMA50
        pullback_zone = self.pullback_atr_mult * atr

        # ── LONG SIGNAL ──────────────────────────────
        if uptrend_bias:
            # Check no duplicate long
            existing_longs = [p for p in current_positions if p.side == 'long']
            if len(existing_longs) >= 1:
                # Allow only if we also allow a short — but max 2 total checked above
                pass

            # Condition 2: Pullback to EMA50 from above
            pullback_to_ema50 = (
                m5_close >= m5_ema50 and
                abs(m5_close - m5_ema50) <= pullback_zone
            )
            # Alternative: previous bar was further from EMA50
            came_from_above = (
                m5_prev_close is not None and
                m5_prev_ema50 is not None and
                (m5_prev_close > m5_prev_ema50 or m5_close <= m5_ema50 + pullback_zone)
            )

            # Condition 3: RSI > 50
            rsi_ok = m5_rsi > 50

            # Condition 4: Bullish candle
            bullish = m5_close > m5_open

            if pullback_to_ema50 and rsi_ok and bullish:
                sl_price = m5_close - (self.sl_atr_mult * atr)
                tp1_price = m5_close + (self.tp1_atr_mult * atr)

                reason = (
                    f"LONG | 1H bias UP (C={h1_close:.2f}>EMA200={h1_ema200:.2f}, "
                    f"EMA50={h1_ema50:.2f}>EMA200) | "
                    f"5M pullback to EMA50={m5_ema50:.2f} (dist={abs(m5_close-m5_ema50):.2f}) | "
                    f"RSI={m5_rsi:.1f} | Bullish candle | ATR={atr:.2f}"
                )
                self.logger.info(f"📈 LONG signal: {reason}")

                return Signal(
                    side='long',
                    entry_price=m5_close,
                    sl_price=round(sl_price, 2),
                    tp1_price=round(tp1_price, 2),
                    atr_value=atr,
                    reason=reason,
                    timestamp=utc_now
                )

        # ── SHORT SIGNAL (exact opposite) ────────────
        if downtrend_bias:
            # Condition 2: Pullback to EMA50 from below
            pullback_to_ema50 = (
                m5_close <= m5_ema50 and
                abs(m5_close - m5_ema50) <= pullback_zone
            )

            # Condition 3: RSI < 50
            rsi_ok = m5_rsi < 50

            # Condition 4: Bearish candle
            bearish = m5_close < m5_open

            if pullback_to_ema50 and rsi_ok and bearish:
                sl_price = m5_close + (self.sl_atr_mult * atr)
                tp1_price = m5_close - (self.tp1_atr_mult * atr)

                reason = (
                    f"SHORT | 1H bias DOWN (C={h1_close:.2f}<EMA200={h1_ema200:.2f}, "
                    f"EMA50={h1_ema50:.2f}<EMA200) | "
                    f"5M pullback to EMA50={m5_ema50:.2f} (dist={abs(m5_close-m5_ema50):.2f}) | "
                    f"RSI={m5_rsi:.1f} | Bearish candle | ATR={atr:.2f}"
                )
                self.logger.info(f"📉 SHORT signal: {reason}")

                return Signal(
                    side='short',
                    entry_price=m5_close,
                    sl_price=round(sl_price, 2),
                    tp1_price=round(tp1_price, 2),
                    atr_value=atr,
                    reason=reason,
                    timestamp=utc_now
                )

        return None

    def check_exit_conditions(
        self,
        position: Position,
        current_price: float,
        current_high: float,
        current_low: float,
        atr: float,
        opposite_signal: bool = False
    ) -> Tuple[Optional[str], float, float]:
        """
        Check SL, TP1, trailing stop, and opposite signal exits.
        
        Returns: (exit_reason, exit_price, close_qty) or (None, 0, 0)
            exit_reason: 'sl', 'tp1', 'trail_stop', 'opposite_signal'
            exit_price: the price at which exit triggered
            close_qty: quantity to close (partial for TP1)
        """
        side = position.side
        entry = position.entry_price
        sl = position.sl_price
        tp1 = position.tp1_price
        remaining = position.remaining_qty
        trail = position.trail_stop

        if side == 'long':
            # ── Stop Loss ──
            if current_low <= sl:
                return ('sl', sl, remaining)

            # ── TP1 (partial close) ──
            if not position.tp1_hit and current_high >= tp1:
                close_qty = round(remaining * self.tp1_close_pct, 8)
                return ('tp1', tp1, close_qty)

            # ── Trailing Stop (after TP1) ──
            if position.tp1_hit:
                # Update highest
                if current_high > position.highest_since_entry:
                    position.highest_since_entry = current_high
                    # Move trail stop up
                    new_trail = position.highest_since_entry - (self.trail_atr_mult * atr)
                    if new_trail > trail:
                        position.trail_stop = new_trail

                if current_low <= position.trail_stop and position.trail_stop > 0:
                    return ('trail_stop', position.trail_stop, remaining)

        elif side == 'short':
            # ── Stop Loss ──
            if current_high >= sl:
                return ('sl', sl, remaining)

            # ── TP1 (partial close) ──
            if not position.tp1_hit and current_low <= tp1:
                close_qty = round(remaining * self.tp1_close_pct, 8)
                return ('tp1', tp1, close_qty)

            # ── Trailing Stop (after TP1) ──
            if position.tp1_hit:
                if current_low < position.lowest_since_entry:
                    position.lowest_since_entry = current_low
                    new_trail = position.lowest_since_entry + (self.trail_atr_mult * atr)
                    if new_trail < trail or trail == 0:
                        position.trail_stop = new_trail

                if current_high >= position.trail_stop and position.trail_stop > 0:
                    return ('trail_stop', position.trail_stop, remaining)

        # ── Opposite Signal ──
        if opposite_signal:
            return ('opposite_signal', current_price, remaining)

        return (None, 0.0, 0.0)
