"""
utils.py — Utilities: config loader, database manager, Telegram alerts, logging setup.
Gold Momentum Multi-TF Scalper Bot.
"""

import json
import logging
import sqlite3
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Represents a single trade for logging."""
    trade_id: str
    symbol: str
    side: str                    # 'long' or 'short'
    entry_price: float
    entry_time: str
    quantity: float
    sl_price: float
    tp1_price: float
    status: str = 'open'        # 'open', 'partial', 'closed'
    exit_price: float = 0.0
    exit_time: str = ''
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ''       # 'tp1', 'tp2_trail', 'sl', 'opposite_signal', 'emergency'
    entry_reason: str = ''
    remaining_qty: float = 0.0
    trail_stop: float = 0.0


@dataclass
class Position:
    """Active in-memory position tracker."""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    entry_time: datetime
    quantity: float
    remaining_qty: float
    sl_price: float
    tp1_price: float
    tp1_hit: bool = False
    trail_stop: float = 0.0
    atr_at_entry: float = 0.0
    highest_since_entry: float = 0.0   # for long trailing
    lowest_since_entry: float = 999999.0  # for short trailing


# ─────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────

class ConfigLoader:
    """Load and validate config.json."""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            cfg = json.load(f)
        self._validate(cfg)
        return cfg

    def _validate(self, cfg: dict):
        required_keys = ['symbol', 'exchange', 'strategy', 'risk', 'backtest', 'database']
        for key in required_keys:
            if key not in cfg:
                raise ValueError(f"Missing required config key: {key}")
        risk = cfg['risk']
        if not (0.1 <= risk['risk_pct'] <= risk.get('max_risk_pct', 1.0)):
            raise ValueError(f"risk_pct must be between 0.1 and {risk.get('max_risk_pct', 1.0)}")

    def get(self, *keys, default=None):
        """Nested key access: config.get('exchange', 'api_key')"""
        val = self.config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def __getitem__(self, key):
        return self.config[key]


# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

def setup_logging(config: ConfigLoader) -> logging.Logger:
    """Configure root logger with console + file handlers."""
    log_level = getattr(logging, config.get('logging', 'level', default='INFO'))
    log_file = config.get('logging', 'file', default='bot.log')

    logger = logging.getLogger('GoldScalper')
    logger.setLevel(log_level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# DATABASE MANAGER (SQLite)
# ─────────────────────────────────────────────

class DatabaseManager:
    """SQLite database for trade logging and PnL tracking."""

    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                quantity REAL NOT NULL,
                sl_price REAL NOT NULL,
                tp1_price REAL NOT NULL,
                status TEXT DEFAULT 'open',
                exit_price REAL DEFAULT 0.0,
                exit_time TEXT DEFAULT '',
                pnl REAL DEFAULT 0.0,
                pnl_pct REAL DEFAULT 0.0,
                exit_reason TEXT DEFAULT '',
                entry_reason TEXT DEFAULT '',
                remaining_qty REAL DEFAULT 0.0,
                trail_stop REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                total_pnl REAL DEFAULT 0.0,
                trade_count INTEGER DEFAULT 0,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                max_drawdown REAL DEFAULT 0.0,
                ending_balance REAL DEFAULT 0.0
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                balance REAL NOT NULL,
                equity REAL NOT NULL,
                drawdown_pct REAL DEFAULT 0.0
            )
        ''')
        self.conn.commit()

    def insert_trade(self, trade: TradeRecord):
        cursor = self.conn.cursor()
        d = asdict(trade)
        cols = ', '.join(d.keys())
        placeholders = ', '.join(['?'] * len(d))
        cursor.execute(f'INSERT OR REPLACE INTO trades ({cols}) VALUES ({placeholders})', list(d.values()))
        self.conn.commit()

    def update_trade(self, trade_id: str, updates: dict):
        cursor = self.conn.cursor()
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        cursor.execute(f'UPDATE trades SET {set_clause} WHERE trade_id = ?',
                       list(updates.values()) + [trade_id])
        self.conn.commit()

    def get_open_trades(self) -> List[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status IN ('open', 'partial')")
        return [dict(row) for row in cursor.fetchall()]

    def get_all_trades(self) -> List[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY entry_time ASC")
        return [dict(row) for row in cursor.fetchall()]

    def get_today_pnl(self) -> float:
        cursor = self.conn.cursor()
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        cursor.execute("SELECT SUM(pnl) FROM trades WHERE entry_time LIKE ? AND status = 'closed'",
                       (f"{today}%",))
        result = cursor.fetchone()
        return result[0] if result[0] else 0.0

    def get_today_trade_count(self) -> int:
        cursor = self.conn.cursor()
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM trades WHERE entry_time LIKE ?", (f"{today}%",))
        result = cursor.fetchone()
        return result[0] if result else 0

    def insert_equity_point(self, balance: float, equity: float, drawdown_pct: float = 0.0):
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO equity_curve (timestamp, balance, equity, drawdown_pct) VALUES (?, ?, ?, ?)',
            (datetime.now(timezone.utc).isoformat(), balance, equity, drawdown_pct)
        )
        self.conn.commit()

    def update_daily_pnl(self, date_str: str, pnl: float, is_win: bool, balance: float):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM daily_pnl WHERE date = ?", (date_str,))
        existing = cursor.fetchone()
        if existing:
            cursor.execute('''
                UPDATE daily_pnl SET 
                    total_pnl = total_pnl + ?,
                    trade_count = trade_count + 1,
                    win_count = win_count + ?,
                    loss_count = loss_count + ?,
                    ending_balance = ?
                WHERE date = ?
            ''', (pnl, 1 if is_win else 0, 0 if is_win else 1, balance, date_str))
        else:
            cursor.execute('''
                INSERT INTO daily_pnl (date, total_pnl, trade_count, win_count, loss_count, ending_balance)
                VALUES (?, ?, 1, ?, ?, ?)
            ''', (date_str, pnl, 1 if is_win else 0, 0 if is_win else 1, balance))
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()


# ─────────────────────────────────────────────
# TELEGRAM ALERTER
# ─────────────────────────────────────────────

class TelegramAlerter:
    """Send trade alerts via Telegram bot API."""

    def __init__(self, config: ConfigLoader, logger: logging.Logger):
        self.enabled = config.get('telegram', 'enabled', default=False)
        self.bot_token = config.get('telegram', 'bot_token', default='')
        self.chat_id = config.get('telegram', 'chat_id', default='')
        self.logger = logger

        if self.enabled and (not self.bot_token or not self.chat_id):
            self.logger.warning("Telegram enabled but token/chat_id missing — disabling.")
            self.enabled = False

    def send(self, message: str):
        """Send a message via Telegram. Non-blocking, errors suppressed."""
        if not self.enabled:
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                self.logger.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
        except Exception as e:
            self.logger.warning(f"Telegram error: {e}")

    def send_trade_open(self, trade: TradeRecord):
        msg = (
            f"🟢 <b>NEW {trade.side.upper()}</b> — {trade.symbol}\n"
            f"Entry: {trade.entry_price:.2f}\n"
            f"SL: {trade.sl_price:.2f}\n"
            f"TP1: {trade.tp1_price:.2f}\n"
            f"Qty: {trade.quantity:.4f}\n"
            f"Reason: {trade.entry_reason}\n"
            f"Time: {trade.entry_time}"
        )
        self.send(msg)

    def send_trade_close(self, trade: TradeRecord):
        emoji = "✅" if trade.pnl > 0 else "🔴"
        msg = (
            f"{emoji} <b>CLOSED {trade.side.upper()}</b> — {trade.symbol}\n"
            f"Entry: {trade.entry_price:.2f} → Exit: {trade.exit_price:.2f}\n"
            f"PnL: {trade.pnl:+.2f} USDT ({trade.pnl_pct:+.2f}%)\n"
            f"Reason: {trade.exit_reason}\n"
            f"Time: {trade.exit_time}"
        )
        self.send(msg)

    def send_alert(self, text: str):
        self.send(f"⚠️ <b>ALERT</b>\n{text}")


# ─────────────────────────────────────────────
# SESSION TIME HELPER
# ─────────────────────────────────────────────

def is_trading_session(config: ConfigLoader, utc_hour: int) -> bool:
    """Check if current UTC hour is within configured London or NY session window."""
    london = config.get('session_hours_utc', 'london', default=[8, 12])
    ny = config.get('session_hours_utc', 'new_york', default=[13, 17])
    return (london[0] <= utc_hour <= london[1]) or (ny[0] <= utc_hour <= ny[1])


# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────

def calculate_position_size(
    balance: float,
    risk_pct: float,
    entry_price: float,
    sl_price: float,
    min_qty: float = 0.001,
    qty_step: float = 0.001
) -> float:
    """
    Calculate position size so that the SL distance risks exactly risk_pct of balance.
    
    Formula: qty = (balance * risk_pct / 100) / abs(entry_price - sl_price)
    """
    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        return 0.0
    
    risk_amount = balance * (risk_pct / 100.0)
    raw_qty = risk_amount / sl_distance
    
    # Round down to qty_step
    qty = max(min_qty, (raw_qty // qty_step) * qty_step)
    return round(qty, 8)


def generate_trade_id(side: str) -> str:
    """Generate unique trade ID."""
    import random
    ts = int(time.time() * 1000)
    rand = random.randint(1000, 9999)
    return f"{side[0].upper()}{ts}_{rand}"


# ─────────────────────────────────────────────
# NEWS AVOIDANCE STUB
# ─────────────────────────────────────────────

class NewsFilter:
    """
    Stub for news-based trade avoidance.
    Extend this to fetch from Forex Factory, Investing.com, or a paid API.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.high_impact_events: List[dict] = []

    def fetch_events(self):
        """Fetch high-impact news events. STUB — implement with real API."""
        self.logger.info("NewsFilter: fetch_events() is a stub — no events loaded.")
        self.high_impact_events = []

    def should_skip_trade(self, utc_now: datetime) -> bool:
        """Return True if a high-impact event is within 30 minutes."""
        # STUB: always returns False
        return False


# ─────────────────────────────────────────────
# DXY CORRELATION FILTER (Optional)
# ─────────────────────────────────────────────

class DXYFilter:
    """
    Optional DXY correlation filter.
    Gold typically moves inverse to DXY. If DXY is rising sharply and we want to go
    long gold, this filter can skip the trade.
    """

    def __init__(self, enabled: bool, logger: logging.Logger):
        self.enabled = enabled
        self.logger = logger
        self.last_dxy_change: float = 0.0

    def update(self, dxy_price: float, dxy_prev: float):
        if dxy_prev > 0:
            self.last_dxy_change = (dxy_price - dxy_prev) / dxy_prev * 100

    def should_skip(self, side: str) -> bool:
        """Skip if DXY is moving in non-correlated direction."""
        if not self.enabled:
            return False
        # Gold is inverse to DXY
        if side == 'long' and self.last_dxy_change > 0.3:
            self.logger.info(f"DXY filter: skipping LONG — DXY up {self.last_dxy_change:.2f}%")
            return True
        if side == 'short' and self.last_dxy_change < -0.3:
            self.logger.info(f"DXY filter: skipping SHORT — DXY down {self.last_dxy_change:.2f}%")
            return True
        return False
