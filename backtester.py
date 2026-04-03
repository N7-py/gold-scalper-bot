"""
backtester.py — Full backtesting engine for Gold Momentum Multi-TF Scalper.

Loads historical 5M and 1H data, simulates trades bar-by-bar, tracks equity curve,
computes performance metrics (win rate, profit factor, max drawdown, Sharpe).
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

from utils import (
    ConfigLoader, Position, TradeRecord, DatabaseManager,
    is_trading_session, calculate_position_size, generate_trade_id
)
from strategy import StrategyEngine, Signal


# ─────────────────────────────────────────────
#  BACKTEST RESULT
# ─────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Summary of a backtest run."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr: float = 0.0
    total_commission: float = 0.0
    initial_balance: float = 0.0
    final_balance: float = 0.0
    return_pct: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    trade_log: List[dict] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
#  DATA LOADER
# ─────────────────────────────────────────────

class DataLoader:
    """Load historical OHLCV data from CSV or Parquet files."""

    @staticmethod
    def load(filepath: str, timeframe: str = '5m') -> pd.DataFrame:
        """
        Load OHLCV data. Expected columns: timestamp/datetime, open, high, low, close, volume.
        Supports CSV and Parquet formats.
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.parquet':
            df = pd.read_parquet(filepath)
        elif ext in ('.csv', '.txt'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Handle timestamp column
        time_col = None
        for candidate in ['timestamp', 'datetime', 'date', 'time', 'open_time']:
            if candidate in df.columns:
                time_col = candidate
                break

        if time_col is None:
            raise ValueError("No timestamp column found. Expected: timestamp, datetime, date, time, or open_time")

        df['timestamp'] = pd.to_datetime(df[time_col], utc=True)
        if time_col != 'timestamp':
            df = df.drop(columns=[time_col])

        # Ensure numeric OHLCV
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                if col == 'volume':
                    df['volume'] = 0
                else:
                    raise ValueError(f"Missing required column: {col}")

        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    @staticmethod
    def resample_to_1h(df_5m: pd.DataFrame) -> pd.DataFrame:
        """Resample 5M data to 1H OHLCV bars."""
        df = df_5m.set_index('timestamp')
        resampled = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        resampled = resampled.reset_index()
        return resampled

    @staticmethod
    def download_from_exchange(
        exchange_name: str,
        symbol: str,
        timeframe: str,
        since: str,
        limit: int = 1000,
        testnet: bool = False
    ) -> pd.DataFrame:
        """Download historical data from exchange via CCXT."""
        import ccxt

        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap' if 'USDT' in symbol else 'spot'}
        })
        if testnet:
            exchange.set_sandbox_mode(True)

        since_ts = exchange.parse8601(since)
        all_ohlcv = []

        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since_ts = ohlcv[-1][0] + 1
            if len(ohlcv) < limit:
                break

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df


# ─────────────────────────────────────────────
#  BACKTESTER
# ─────────────────────────────────────────────

class Backtester:
    """
    Bar-by-bar backtesting engine.
    Simulates the exact strategy rules on historical data.
    """

    def __init__(self, config: ConfigLoader, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.strategy = StrategyEngine(config, logger)

        # Backtest params
        self.initial_balance = config.get('backtest', 'initial_balance', default=10000.0)
        self.commission_pct = config.get('backtest', 'commission_pct', default=0.04) / 100.0
        self.slippage_pct = config.get('backtest', 'slippage_pct', default=0.01) / 100.0
        self.risk_pct = config.get('risk', 'risk_pct', default=0.5)
        self.max_daily_dd = config.get('risk', 'max_daily_drawdown_pct', default=5.0)

    def run(
        self,
        df_5m: pd.DataFrame,
        df_1h: Optional[pd.DataFrame] = None,
        save_db: bool = True
    ) -> BacktestResult:
        """
        Run full backtest on provided data.
        
        Args:
            df_5m: 5-minute OHLCV DataFrame
            df_1h: 1-hour OHLCV DataFrame (auto-resampled from 5M if not provided)
            save_db: whether to save trades to database
        """
        self.logger.info("=" * 60)
        self.logger.info("BACKTESTER STARTING")
        self.logger.info("=" * 60)

        # ── Prepare data ─────────────────────────────
        if df_1h is None:
            self.logger.info("Resampling 5M data to 1H...")
            df_1h = DataLoader.resample_to_1h(df_5m)

        # Compute indicators
        self.logger.info("Computing indicators on 5M data...")
        df_5m = self.strategy.compute_indicators(df_5m)
        self.logger.info("Computing indicators on 1H data...")
        df_1h = self.strategy.compute_indicators(df_1h)

        # ── State ────────────────────────────────────
        balance = self.initial_balance
        peak_balance = balance
        positions: List[Position] = []
        closed_trades: List[TradeRecord] = []
        equity_curve = [balance]
        drawdown_curve = [0.0]
        timestamps = [df_5m.iloc[0]['timestamp'].isoformat()]
        total_commission = 0.0

        # Daily tracking for emergency stop
        current_day = None
        day_start_balance = balance

        # Trade cooldown: minimum bars between entries (prevents over-trading)
        last_entry_bar = -999
        min_bars_between_trades = 6  # 30 minutes cooldown (6 x 5M bars)

        # Need enough bars for indicators (EMA200 needs 200 bars)
        lookback_1h = 210
        lookback_5m = 210
        start_idx = lookback_5m

        total_bars = len(df_5m)
        self.logger.info(f"Total 5M bars: {total_bars}, starting from bar {start_idx}")
        self.logger.info(f"Initial balance: ${self.initial_balance:,.2f}")
        self.logger.info(f"Risk per trade: {self.risk_pct}%")

        # ── Bar-by-bar simulation ────────────────
        for i in range(start_idx, total_bars):
            bar = df_5m.iloc[i]
            bar_time = bar['timestamp']
            bar_close = bar['close']
            bar_high = bar['high']
            bar_low = bar['low']
            bar_open = bar['open']

            # ── Daily drawdown check ─────────────
            bar_date = bar_time.strftime('%Y-%m-%d')
            if current_day != bar_date:
                current_day = bar_date
                day_start_balance = balance

            daily_dd = (day_start_balance - balance) / day_start_balance * 100 if day_start_balance > 0 else 0
            if daily_dd >= self.max_daily_dd:
                # Emergency stop — close all positions
                for pos in list(positions):
                    exit_price = bar_close
                    pnl = self._calc_pnl(pos, exit_price, pos.remaining_qty)
                    commission = abs(pnl) * self.commission_pct + 0.50  # Flat + % of PnL
                    total_commission += commission
                    pnl -= commission
                    balance += pnl

                    trade_rec = self._close_trade_record(
                        pos, exit_price, bar_time, pnl, balance, 'emergency_dd_stop'
                    )
                    closed_trades.append(trade_rec)
                    self.logger.warning(
                        f"🚨 EMERGENCY STOP — daily DD {daily_dd:.2f}% — closing {pos.side} @ {exit_price:.2f}"
                    )
                positions.clear()
                continue  # Skip rest of this bar

            # ── Check exits on current positions ──
            atr = bar['atr'] if not pd.isna(bar['atr']) else 0
            positions_to_remove = []
            
            for pos in positions:
                # Check for opposite signal (will be evaluated after exit checks)
                has_opposite = False

                exit_reason, exit_price, close_qty = self.strategy.check_exit_conditions(
                    pos, bar_close, bar_high, bar_low, atr, has_opposite
                )

                if exit_reason:
                    # Apply slippage
                    if pos.side == 'long':
                        exit_price = exit_price * (1 - self.slippage_pct)
                    else:
                        exit_price = exit_price * (1 + self.slippage_pct)

                    pnl = self._calc_pnl(pos, exit_price, close_qty)
                    commission = abs(pnl) * self.commission_pct + 0.25
                    total_commission += commission
                    pnl -= commission
                    balance += pnl

                    if exit_reason == 'tp1':
                        # Partial close
                        pos.remaining_qty -= close_qty
                        pos.tp1_hit = True
                        # Initialize trailing
                        if pos.side == 'long':
                            pos.highest_since_entry = bar_high
                            pos.trail_stop = bar_close - (self.strategy.trail_atr_mult * atr)
                        else:
                            pos.lowest_since_entry = bar_low
                            pos.trail_stop = bar_close + (self.strategy.trail_atr_mult * atr)

                        self.logger.debug(
                            f"  TP1 hit: {pos.side} partial close {close_qty:.4f} @ {exit_price:.2f} "
                            f"| PnL: {pnl:+.2f} | Remaining: {pos.remaining_qty:.4f}"
                        )
                    else:
                        # Full close
                        trade_rec = self._close_trade_record(
                            pos, exit_price, bar_time, pnl, balance, exit_reason
                        )
                        closed_trades.append(trade_rec)
                        positions_to_remove.append(pos)

                        self.logger.debug(
                            f"  CLOSED {pos.side} @ {exit_price:.2f} reason={exit_reason} "
                            f"| PnL: {pnl:+.2f} | Balance: {balance:,.2f}"
                        )

            for pos in positions_to_remove:
                positions.remove(pos)

            # ── Evaluate new signal ──────────────
            # Get the 1H bar that corresponds to this 5M bar
            h1_mask = df_1h['timestamp'] <= bar_time
            if h1_mask.sum() < lookback_1h:
                equity_curve.append(balance)
                drawdown_curve.append(0.0)
                timestamps.append(bar_time.isoformat())
                continue

            # Window of recent 1H bars
            df_1h_window = df_1h[h1_mask].tail(lookback_1h + 50)
            df_5m_window = df_5m.iloc[max(0, i - lookback_5m - 50):i + 1]

            signal = self.strategy.evaluate(
                df_1h_window,
                df_5m_window,
                positions,
                bar_time.to_pydatetime().replace(tzinfo=timezone.utc) if bar_time.tzinfo is None else bar_time.to_pydatetime()
            )

            if signal and (i - last_entry_bar) >= min_bars_between_trades:
                # Apply slippage to entry
                entry_price = signal.entry_price
                if signal.side == 'long':
                    entry_price = entry_price * (1 + self.slippage_pct)
                else:
                    entry_price = entry_price * (1 - self.slippage_pct)

                # Calculate position size
                qty = calculate_position_size(
                    balance, self.risk_pct, entry_price, signal.sl_price
                )

                risk_amount = abs(entry_price - signal.sl_price) * qty
                if qty > 0 and risk_amount <= balance * (self.risk_pct / 100.0) * 1.1:
                    # Entry commission (flat fee model)
                    commission = 0.50  # fixed per-trade commission
                    total_commission += commission
                    balance -= commission
                    last_entry_bar = i

                    trade_id = generate_trade_id(signal.side)
                    new_pos = Position(
                        trade_id=trade_id,
                        symbol=self.config['symbol'],
                        side=signal.side,
                        entry_price=entry_price,
                        entry_time=bar_time.to_pydatetime(),
                        quantity=qty,
                        remaining_qty=qty,
                        sl_price=signal.sl_price,
                        tp1_price=signal.tp1_price,
                        atr_at_entry=signal.atr_value,
                        highest_since_entry=entry_price if signal.side == 'long' else 0,
                        lowest_since_entry=entry_price if signal.side == 'short' else 999999
                    )
                    positions.append(new_pos)

                    self.logger.debug(
                        f"  ENTRY {signal.side.upper()} @ {entry_price:.2f} | "
                        f"SL: {signal.sl_price:.2f} | TP1: {signal.tp1_price:.2f} | "
                        f"Qty: {qty:.4f} | Balance: {balance:,.2f}"
                    )

            # ── Update equity curve ──────────────
            unrealized_pnl = sum(
                self._calc_pnl(p, bar_close, p.remaining_qty) for p in positions
            )
            equity = balance + unrealized_pnl
            peak_balance = max(peak_balance, equity)
            dd = (peak_balance - equity) / peak_balance * 100 if peak_balance > 0 else 0

            equity_curve.append(equity)
            drawdown_curve.append(dd)
            timestamps.append(bar_time.isoformat())

            # Progress logging every 10,000 bars
            if (i - start_idx) % 10000 == 0 and i > start_idx:
                self.logger.info(
                    f"  Progress: {i}/{total_bars} bars | "
                    f"Balance: ${balance:,.2f} | Positions: {len(positions)} | "
                    f"Trades: {len(closed_trades)}"
                )

        # ── Close remaining positions at last price ──
        last_bar = df_5m.iloc[-1]
        for pos in positions:
            exit_price = last_bar['close']
            pnl = self._calc_pnl(pos, exit_price, pos.remaining_qty)
            commission = abs(pnl) * self.commission_pct + 0.50
            total_commission += commission
            pnl -= commission
            balance += pnl
            trade_rec = self._close_trade_record(
                pos, exit_price, last_bar['timestamp'], pnl, balance, 'backtest_end'
            )
            closed_trades.append(trade_rec)

        # ── Compute metrics ──────────────────────
        result = self._compute_metrics(closed_trades, equity_curve, drawdown_curve,
                                        timestamps, total_commission)
        result.initial_balance = self.initial_balance
        result.final_balance = balance

        # ── Save to DB ───────────────────────────
        if save_db:
            db = DatabaseManager(self.config.get('database', 'path', default='backtest_trades.db'))
            for tr in closed_trades:
                db.insert_trade(tr)
            db.close()
            self.logger.info(f"Trades saved to database.")

        # ── Print summary ────────────────────────
        self._print_summary(result)

        return result

    def _calc_pnl(self, pos: Position, exit_price: float, qty: float) -> float:
        """Calculate PnL for a position."""
        if pos.side == 'long':
            return (exit_price - pos.entry_price) * qty
        else:
            return (pos.entry_price - exit_price) * qty

    def _close_trade_record(
        self, pos: Position, exit_price: float, exit_time,
        pnl: float, balance: float, reason: str
    ) -> TradeRecord:
        """Create a TradeRecord for a closed trade."""
        pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100 if pos.entry_price * pos.quantity > 0 else 0
        exit_time_str = exit_time.isoformat() if hasattr(exit_time, 'isoformat') else str(exit_time)
        entry_time_str = pos.entry_time.isoformat() if hasattr(pos.entry_time, 'isoformat') else str(pos.entry_time)

        return TradeRecord(
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            entry_time=entry_time_str,
            quantity=pos.quantity,
            sl_price=pos.sl_price,
            tp1_price=pos.tp1_price,
            status='closed',
            exit_price=exit_price,
            exit_time=exit_time_str,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 4),
            exit_reason=reason,
            remaining_qty=0,
            trail_stop=pos.trail_stop
        )

    def _compute_metrics(
        self,
        trades: List[TradeRecord],
        equity_curve: List[float],
        drawdown_curve: List[float],
        timestamps: List[str],
        total_commission: float
    ) -> BacktestResult:
        """Compute all backtest performance metrics."""
        result = BacktestResult()
        result.equity_curve = equity_curve
        result.drawdown_curve = drawdown_curve
        result.timestamps = timestamps
        result.total_commission = total_commission
        result.trade_log = [asdict(t) for t in trades]

        result.total_trades = len(trades)
        if result.total_trades == 0:
            return result

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = (result.winning_trades / result.total_trades) * 100

        result.gross_profit = sum(t.pnl for t in wins)
        result.gross_loss = abs(sum(t.pnl for t in losses))
        result.net_profit = result.gross_profit - result.gross_loss

        result.profit_factor = (
            result.gross_profit / result.gross_loss if result.gross_loss > 0 else float('inf')
        )

        result.avg_win = result.gross_profit / len(wins) if wins else 0
        result.avg_loss = result.gross_loss / len(losses) if losses else 0
        result.avg_rr = result.avg_win / result.avg_loss if result.avg_loss > 0 else float('inf')

        # Max drawdown
        result.max_drawdown_pct = max(drawdown_curve) if drawdown_curve else 0
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd_abs = peak - eq
        result.max_drawdown = float(np.max(dd_abs))

        # Sharpe ratio (daily returns)
        if len(equity_curve) > 1:
            eq_series = pd.Series(equity_curve)
            returns = eq_series.pct_change().dropna()
            if returns.std() > 0:
                # Annualized (assuming 5M bars: ~288 bars/day, ~252 trading days)
                result.sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(288 * 252))
            else:
                result.sharpe_ratio = 0.0

        result.return_pct = (result.net_profit / self.initial_balance) * 100

        return result

    def _print_summary(self, result: BacktestResult):
        """Print a formatted backtest summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("             BACKTEST RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"  Period:            {result.timestamps[0][:10]} → {result.timestamps[-1][:10]}")
        self.logger.info(f"  Initial Balance:   ${result.initial_balance:>12,.2f}")
        self.logger.info(f"  Final Balance:     ${result.final_balance:>12,.2f}")
        self.logger.info(f"  Net Profit:        ${result.net_profit:>12,.2f} ({result.return_pct:+.2f}%)")
        self.logger.info(f"  Total Commission:  ${result.total_commission:>12,.2f}")
        self.logger.info("-" * 60)
        self.logger.info(f"  Total Trades:      {result.total_trades:>6}")
        self.logger.info(f"  Winning Trades:    {result.winning_trades:>6}")
        self.logger.info(f"  Losing Trades:     {result.losing_trades:>6}")
        self.logger.info(f"  Win Rate:          {result.win_rate:>6.1f}%")
        self.logger.info("-" * 60)
        self.logger.info(f"  Gross Profit:      ${result.gross_profit:>12,.2f}")
        self.logger.info(f"  Gross Loss:        ${result.gross_loss:>12,.2f}")
        self.logger.info(f"  Profit Factor:     {result.profit_factor:>8.2f}")
        self.logger.info(f"  Avg Win:           ${result.avg_win:>12,.2f}")
        self.logger.info(f"  Avg Loss:          ${result.avg_loss:>12,.2f}")
        self.logger.info(f"  Avg R:R:           {result.avg_rr:>8.2f}")
        self.logger.info("-" * 60)
        self.logger.info(f"  Max Drawdown:      ${result.max_drawdown:>12,.2f} ({result.max_drawdown_pct:.2f}%)")
        self.logger.info(f"  Sharpe Ratio:      {result.sharpe_ratio:>8.2f}")
        self.logger.info("=" * 60)

    def plot_equity_curve(self, result: BacktestResult, save_path: str = "equity_curve.png"):
        """Generate and save equity curve + drawdown chart."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]},
                sharex=True
            )
            fig.suptitle('Gold Momentum Multi-TF Scalper — Backtest Results',
                         fontsize=14, fontweight='bold')

            # Parse timestamps (sample every N points for performance)
            n_points = len(result.equity_curve)
            step = max(1, n_points // 2000)
            indices = range(0, n_points, step)
            
            eq_sampled = [result.equity_curve[i] for i in indices]
            dd_sampled = [result.drawdown_curve[i] for i in indices]
            x = list(range(len(eq_sampled)))

            # Equity curve
            ax1.plot(x, eq_sampled, color='#00C853', linewidth=1.2, label='Equity')
            ax1.axhline(y=result.initial_balance, color='gray', linestyle='--',
                       alpha=0.5, label='Initial Balance')
            ax1.fill_between(x, result.initial_balance, eq_sampled,
                            where=[e >= result.initial_balance for e in eq_sampled],
                            alpha=0.15, color='#00C853')
            ax1.fill_between(x, result.initial_balance, eq_sampled,
                            where=[e < result.initial_balance for e in eq_sampled],
                            alpha=0.15, color='#FF1744')
            ax1.set_ylabel('Equity ($)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Add summary text box
            stats_text = (
                f"Net: ${result.net_profit:+,.2f} ({result.return_pct:+.1f}%)  |  "
                f"Trades: {result.total_trades}  |  WR: {result.win_rate:.1f}%  |  "
                f"PF: {result.profit_factor:.2f}  |  MaxDD: {result.max_drawdown_pct:.1f}%"
            )
            ax1.set_title(stats_text, fontsize=10, color='gray')

            # Drawdown
            ax2.fill_between(x, 0, [-d for d in dd_sampled], color='#FF1744', alpha=0.4)
            ax2.plot(x, [-d for d in dd_sampled], color='#FF1744', linewidth=0.8)
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Bar Index (sampled)')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Equity curve saved to: {save_path}")

        except ImportError:
            self.logger.warning("matplotlib not available — cannot plot equity curve.")

    def export_trades_csv(self, result: BacktestResult, filepath: str = "backtest_trades.csv"):
        """Export trade log to CSV."""
        if not result.trade_log:
            self.logger.warning("No trades to export.")
            return
        df = pd.DataFrame(result.trade_log)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Trade log exported to: {filepath}")
