"""
server.py — Koyeb deployment entry point.
Runs the trading bot alongside a lightweight HTTP health-check server.

All sensitive config is read from environment variables:
  EXCHANGE_NAME       (default: binance)
  EXCHANGE_API_KEY    (required)
  EXCHANGE_API_SECRET (required)
  EXCHANGE_TESTNET    (default: false)
  BOT_MODE            (default: paper)   — 'paper' or 'live'
  RISK_PCT            (default: 0.5)
  TELEGRAM_ENABLED    (default: false)
  TELEGRAM_BOT_TOKEN  (optional)
  TELEGRAM_CHAT_ID    (optional)
  PORT                (default: 8000)    — Koyeb sets this automatically
"""

import asyncio
import json
import os
import signal
import sys
import threading
import time
import logging
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Build config from env vars ──────────────────────────────────────────────

def build_config_from_env() -> dict:
    """Build config dict from environment variables, falling back to config.json."""
    
    # Start from config.json if it exists
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with env vars
    mode = os.getenv("BOT_MODE", "paper")
    
    config["symbol"] = os.getenv("SYMBOL", config.get("symbol", "XAU/USDT:USDT"))
    config["market_type"] = os.getenv("MARKET_TYPE", config.get("market_type", "swap"))
    config["mode"] = mode
    
    # Exchange
    config.setdefault("exchange", {})
    config["exchange"]["name"] = os.getenv("EXCHANGE_NAME", config["exchange"].get("name", "binance"))
    config["exchange"]["api_key"] = os.getenv("EXCHANGE_API_KEY", config["exchange"].get("api_key", ""))
    config["exchange"]["api_secret"] = os.getenv("EXCHANGE_API_SECRET", config["exchange"].get("api_secret", ""))
    config["exchange"]["api_password"] = os.getenv("EXCHANGE_API_PASSWORD", config["exchange"].get("api_password", ""))
    config["exchange"]["testnet"] = os.getenv("EXCHANGE_TESTNET", "false").lower() == "true"
    
    # Risk
    config.setdefault("risk", {})
    config["risk"]["risk_pct"] = float(os.getenv("RISK_PCT", config["risk"].get("risk_pct", 0.5)))
    config["risk"]["max_risk_pct"] = float(os.getenv("MAX_RISK_PCT", config["risk"].get("max_risk_pct", 1.0)))
    config["risk"]["max_concurrent_positions"] = int(os.getenv("MAX_POSITIONS", config["risk"].get("max_concurrent_positions", 2)))
    config["risk"]["max_daily_drawdown_pct"] = float(os.getenv("MAX_DAILY_DD", config["risk"].get("max_daily_drawdown_pct", 5.0)))
    config["risk"]["min_atr_threshold"] = float(os.getenv("MIN_ATR", config["risk"].get("min_atr_threshold", 0.5)))
    
    # Strategy (keep defaults from config.json or use sensible defaults)
    config.setdefault("strategy", {})
    config["strategy"].setdefault("ema_fast", 50)
    config["strategy"].setdefault("ema_slow", 200)
    config["strategy"].setdefault("rsi_period", 14)
    config["strategy"].setdefault("atr_period", 14)
    config["strategy"].setdefault("pullback_atr_mult", 2.0)
    config["strategy"].setdefault("sl_atr_mult", 1.5)
    config["strategy"].setdefault("tp1_atr_mult", 2.0)
    config["strategy"].setdefault("trail_atr_mult", 1.0)
    config["strategy"].setdefault("tp1_close_pct", 0.5)
    config["strategy"].setdefault("short_sl_atr_mult", 2.0)
    config["strategy"].setdefault("short_tp1_atr_mult", 3.0)
    config["strategy"].setdefault("rsi_bear_range_period", 10)
    config["strategy"].setdefault("rsi_bear_range_max", 65.0)

    # Session hours — wide windows to cover London + NY fully (XAUT trades 24/7)
    config.setdefault("session_hours_utc", {"london": [7, 16], "new_york": [13, 21]})
    
    # Execution
    config.setdefault("execution", {"order_type": "market", "use_oco": False})
    
    # Backtest
    config.setdefault("backtest", {
        "data_dir": "data",
        "initial_balance": float(os.getenv("INITIAL_BALANCE", 10000.0)),
        "commission_pct": 0.04,
        "slippage_pct": 0.01
    })
    
    # Database (use /tmp for Koyeb ephemeral storage)
    config.setdefault("database", {})
    config["database"]["type"] = "sqlite"
    config["database"]["path"] = os.getenv("DB_PATH", "/tmp/trades.db")
    
    # Telegram
    config.setdefault("telegram", {})
    config["telegram"]["enabled"] = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
    config["telegram"]["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN", config["telegram"].get("bot_token", ""))
    config["telegram"]["chat_id"] = os.getenv("TELEGRAM_CHAT_ID", config["telegram"].get("chat_id", ""))
    
    # Logging
    config.setdefault("logging", {})
    config["logging"]["level"] = os.getenv("LOG_LEVEL", "INFO")
    config["logging"]["file"] = "/tmp/bot.log"
    
    return config


# ── Health check HTTP server ────────────────────────────────────────────────

class HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for Koyeb health checks + bot status."""

    bot_instance = None  # Set by main thread
    start_time = time.time()
    _cache: dict = {}
    _CACHE_TTL = 10  # seconds

    @classmethod
    def _get_cached(cls, key):
        entry = cls._cache.get(key)
        if entry and (time.time() - entry[0]) < cls._CACHE_TTL:
            return entry[1]
        return None

    @classmethod
    def _set_cached(cls, key, data):
        cls._cache[key] = (time.time(), data)

    def do_GET(self):
        if self.path == "/":
            self._respond_index()
        elif self.path == "/health":
            self._respond_health()
        elif self.path == "/status":
            self._respond_status()
        elif self.path.startswith("/candles"):
            self._respond_candles()
        elif self.path == "/conditions":
            self._respond_conditions()
        elif self.path == "/trades":
            self._respond_trades()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/mode":
            self._handle_set_mode()
        elif self.path == "/force_trade":
            self._handle_force_trade()
        else:
            self.send_error(404)

    def _handle_set_mode(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            new_mode = body.get("mode", "").lower()
        except Exception:
            self._json_response(400, {"error": "Invalid JSON body"})
            return

        if new_mode not in ("paper", "live"):
            self._json_response(400, {"error": "mode must be 'paper' or 'live'"})
            return

        bot = self.bot_instance
        if not bot:
            self._json_response(503, {"error": "Bot not running"})
            return

        if new_mode == bot.mode:
            self._json_response(200, {"mode": bot.mode, "changed": False})
            return

        if new_mode == "live":
            api_key = bot.config.get("exchange", "api_key", default="")
            if not api_key:
                self._json_response(400, {"error": "EXCHANGE_API_KEY not configured — cannot switch to live"})
                return
            try:
                bot.exchange._init_exchange()
                bot.exchange.paper_mode = False
            except Exception as e:
                self._json_response(500, {"error": f"Exchange init failed: {e}"})
                return
        else:
            bot.exchange.paper_mode = True
            bot.exchange.exchange = None
            bot.exchange._init_public_exchange()

        bot.mode = new_mode
        self._json_response(200, {"mode": new_mode, "changed": True})

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
            
    def _respond_index(self):
        index_path = os.path.join(os.path.dirname(__file__), "index.html")
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, "Dashboard not found")
    
    def _respond_health(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        uptime = int(time.time() - self.start_time)
        body = json.dumps({
            "status": "healthy",
            "service": "gold-scalper-bot",
            "uptime_seconds": uptime,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        self.wfile.write(body.encode())
    
    def _respond_status(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        
        bot = self.bot_instance
        status = {
            "service": "gold-scalper-bot",
            "uptime_seconds": int(time.time() - self.start_time),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bot": {
                "running": bot.running if bot else False,
                "mode": bot.mode if bot else "unknown",
                "symbol": bot.symbol if bot else "unknown",
                "positions": len(bot.positions) if bot else 0,
                "emergency_stop": bot.emergency_stop if bot else False,
                "balance": round(bot.exchange.get_balance(), 2) if bot and bot.exchange else 0,
            } if bot else {"running": False}
        }
        self.wfile.write(json.dumps(status, indent=2).encode())
    
    def _respond_candles(self):
        cached = self._get_cached('candles')
        if cached:
            self._json_response(200, cached)
            return

        bot = self.bot_instance
        if not bot or not bot.exchange:
            self._json_response(503, {"error": "Bot not ready"})
            return

        try:
            import pandas as pd
            df = bot.exchange.fetch_ohlcv(bot.symbol, '5m', limit=120)
            if df.empty:
                self._json_response(503, {"error": "No candle data"})
                return

            df = bot.strategy.indicators.compute(df)

            candles, ema50_data, ema200_data = [], [], []
            for _, row in df.iterrows():
                t = int(row['timestamp'].timestamp())
                candles.append({
                    'time': t,
                    'open': round(float(row['open']), 2),
                    'high': round(float(row['high']), 2),
                    'low': round(float(row['low']), 2),
                    'close': round(float(row['close']), 2),
                })
                if not pd.isna(row['ema50']):
                    ema50_data.append({'time': t, 'value': round(float(row['ema50']), 2)})
                if not pd.isna(row['ema200']):
                    ema200_data.append({'time': t, 'value': round(float(row['ema200']), 2)})

            data = {'candles': candles, 'ema50': ema50_data, 'ema200': ema200_data}
            self._set_cached('candles', data)
            self._json_response(200, data)
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _respond_conditions(self):
        cached = self._get_cached('conditions')
        if cached:
            self._json_response(200, cached)
            return

        bot = self.bot_instance
        if not bot or not bot.exchange:
            self._json_response(503, {"error": "Bot not ready"})
            return

        try:
            import pandas as pd
            from utils import is_trading_session

            utc_now = datetime.now(timezone.utc)
            symbol = bot.symbol

            df_5m = bot.exchange.fetch_ohlcv(symbol, '5m', limit=250)
            df_1h = bot.exchange.fetch_ohlcv(symbol, '1h', limit=250)

            if df_5m.empty or df_1h.empty:
                self._json_response(503, {"error": "No market data"})
                return

            df_5m = bot.strategy.indicators.compute(df_5m)
            df_1h = bot.strategy.indicators.compute(df_1h)

            bar_5m = df_5m.iloc[-1]
            bar_1h = df_1h.iloc[-1]

            price      = round(float(bar_5m['close']), 2)
            m5_open    = round(float(bar_5m['open']), 2)
            m5_ema50   = round(float(bar_5m['ema50']), 2)      if not pd.isna(bar_5m['ema50'])      else None
            m5_rsi     = round(float(bar_5m['rsi']), 1)        if not pd.isna(bar_5m['rsi'])        else None
            atr        = round(float(bar_5m['atr']), 2)        if not pd.isna(bar_5m['atr'])        else None
            macd_hist  = round(float(bar_5m['macd_hist']), 4)  if not pd.isna(bar_5m['macd_hist'])  else None
            stochrsi_k = round(float(bar_5m['stochrsi_k']), 1) if not pd.isna(bar_5m['stochrsi_k']) else None

            h1_close  = round(float(bar_1h['close']), 2)
            h1_ema50  = round(float(bar_1h['ema50']), 2)  if not pd.isna(bar_1h['ema50'])  else None
            h1_ema200 = round(float(bar_1h['ema200']), 2) if not pd.isna(bar_1h['ema200']) else None

            min_atr       = bot.strategy.min_atr
            pullback_mult = bot.strategy.pullback_atr_mult
            pullback_zone = round(pullback_mult * atr, 2) if atr else None

            # RSI Bear Range (short-specific) — RSI max over last N bars
            rsi_br_period = bot.strategy.rsi_bear_range_period
            rsi_br_max    = bot.strategy.rsi_bear_range_max
            rsi_window    = df_5m['rsi'].iloc[-(rsi_br_period + 1):-1].dropna()
            rsi_br_value  = round(float(rsi_window.max()), 1) if len(rsi_window) > 0 else None
            rsi_br_ok     = rsi_br_value is not None and rsi_br_value < rsi_br_max

            # ── Session ──
            utc_hour   = utc_now.hour
            session_ok = is_trading_session(bot.config, utc_hour)
            london     = bot.config.get('session_hours_utc', 'london',   default=[8, 12])
            ny         = bot.config.get('session_hours_utc', 'new_york', default=[13, 17])
            if london[0] <= utc_hour < london[1]:
                session_val = f"London ({utc_hour:02d}:xx UTC)"
            elif ny[0] <= utc_hour < ny[1]:
                session_val = f"New York ({utc_hour:02d}:xx UTC)"
            else:
                session_val = f"Closed ({utc_hour:02d}:xx UTC)"

            # ── 1H Bias ──
            uptrend   = h1_ema200 is not None and h1_close > h1_ema200
            downtrend = h1_ema200 is not None and h1_close < h1_ema200
            bias = "long" if uptrend else ("short" if downtrend else "neutral")

            h1_bias_val = (
                f"UP — Close {h1_close} > EMA200 {h1_ema200}" if uptrend else
                f"DOWN — Close {h1_close} < EMA200 {h1_ema200}" if downtrend else
                f"NEUTRAL — Close {h1_close} / EMA200 {h1_ema200}"
            )
            ema_cross_val = f"EMA50 {h1_ema50} {'>' if uptrend else '<'} EMA200 {h1_ema200}" if h1_ema50 and h1_ema200 else "N/A"

            # ── ATR filter ──
            atr_ok  = atr is not None and atr >= min_atr
            atr_val = f"{atr} (min {min_atr})" if atr else "N/A"

            # ── Shared values ──
            bullish       = price > m5_open
            n_pos         = len(bot.positions)
            max_pos       = bot.max_positions
            pos_ok        = n_pos < max_pos
            pos_val       = f"{n_pos} / {max_pos} open"
            pullback_dist = round(abs(price - m5_ema50), 2) if m5_ema50 else None
            pullback_val  = f"Dist {pullback_dist} (zone ≤ {pullback_zone})" if pullback_dist is not None else "N/A"
            macd_ok       = macd_hist is not None and macd_hist < 0
            macd_val      = f"{macd_hist}" if macd_hist is not None else "N/A"
            rsi_val       = str(m5_rsi) if m5_rsi else "N/A"
            stochrsi_val  = str(stochrsi_k) if stochrsi_k is not None else "N/A"
            rsi_br_val    = (f"10-bar max={rsi_br_value} (need <{rsi_br_max})"
                             if rsi_br_value is not None else "N/A")

            # ── LONG conditions (evaluated independently) ──
            long_pullback_ok = (m5_ema50 is not None and pullback_zone is not None
                                and pullback_dist <= pullback_zone)
            long_rsi_ok      = m5_rsi is not None and m5_rsi > 45
            long_candle_ok   = bullish  # informational only
            long_candle_val  = (f"Bullish — C {price} > O {m5_open}" if bullish
                                else f"Bearish — C {price} < O {m5_open}")
            long_h1_val      = (f"UP — Close {h1_close} > EMA200 {h1_ema200}" if uptrend
                                else f"NOT UP — Close {h1_close} / EMA200 {h1_ema200}")
            long_cross_val   = (f"EMA50 {h1_ema50} > EMA200 {h1_ema200}" if h1_ema50 and h1_ema200
                                else "N/A")
            long_signal_ready = all([session_ok, atr_ok, uptrend,
                                     long_pullback_ok, long_rsi_ok, pos_ok])

            long_conditions = [
                {"id": "session",   "label": "Trading Session",      "pass": session_ok,       "value": session_val,    "required": "London 7–16 / NY 13–21 UTC"},
                {"id": "atr",       "label": "ATR Filter (5M)",      "pass": atr_ok,           "value": atr_val,        "required": f"≥ {min_atr}"},
                {"id": "h1_bias",   "label": "1H Trend Bias",        "pass": uptrend,          "value": long_h1_val,    "required": "Close > EMA200 (bullish structure)"},
                {"id": "h1_cross",  "label": "1H EMA50 vs EMA200 (info)", "pass": uptrend,      "value": long_cross_val, "required": "EMA50 > EMA200 (informational)"},
                {"id": "pullback",  "label": "5M Pullback to EMA50", "pass": long_pullback_ok, "value": pullback_val,   "required": f"Price ≥ EMA50, within {pullback_mult}×ATR"},
                {"id": "rsi",       "label": "5M RSI(14)",            "pass": long_rsi_ok,      "value": rsi_val,        "required": "> 45 (bullish momentum)"},
                {"id": "candle",    "label": "Candle Direction (info)", "pass": long_candle_ok, "value": long_candle_val,"required": "Bullish preferred (informational)"},
                {"id": "positions", "label": "Position Limit",        "pass": pos_ok,           "value": pos_val,        "required": f"< {max_pos} concurrent"},
            ]

            # ── SHORT conditions (evaluated independently, includes enhanced filters) ──
            short_pullback_ok = (m5_ema50 is not None and pullback_zone is not None
                                 and pullback_dist <= pullback_zone)
            short_rsi_ok      = m5_rsi is not None and m5_rsi < 55
            short_candle_ok   = not bullish  # informational only
            short_candle_val  = (f"Bearish — C {price} < O {m5_open}" if not bullish
                                 else f"Bullish — C {price} > O {m5_open}")
            short_h1_val      = (f"DOWN — Close {h1_close} < EMA200 {h1_ema200}" if downtrend
                                 else f"NOT DOWN — Close {h1_close} / EMA200 {h1_ema200}")
            short_cross_val   = (f"EMA50 {h1_ema50} < EMA200 {h1_ema200}" if h1_ema50 and h1_ema200
                                 else "N/A")
            short_signal_ready = all([session_ok, atr_ok, downtrend,
                                      short_pullback_ok, short_rsi_ok,
                                      pos_ok, rsi_br_ok])

            short_conditions = [
                {"id": "session",        "label": "Trading Session",          "pass": session_ok,        "value": session_val,    "required": "London 7–16 / NY 13–21 UTC"},
                {"id": "atr",            "label": "ATR Filter (5M)",          "pass": atr_ok,            "value": atr_val,        "required": f"≥ {min_atr}"},
                {"id": "h1_bias",        "label": "1H Trend Bias",            "pass": downtrend,         "value": short_h1_val,   "required": "Close < EMA200 (bearish structure)"},
                {"id": "h1_cross",       "label": "1H EMA50 vs EMA200 (info)", "pass": downtrend,        "value": short_cross_val,"required": "EMA50 < EMA200 (informational)"},
                {"id": "pullback",       "label": "5M Pullback to EMA50",     "pass": short_pullback_ok, "value": pullback_val,   "required": f"Price ≤ EMA50, within {pullback_mult}×ATR"},
                {"id": "rsi",            "label": "5M RSI(14)",                "pass": short_rsi_ok,      "value": rsi_val,        "required": "< 55 (bearish momentum)"},
                {"id": "candle",         "label": "Candle Direction (info)",   "pass": short_candle_ok,   "value": short_candle_val,"required": "Bearish preferred (informational)"},
                {"id": "positions",      "label": "Position Limit",            "pass": pos_ok,            "value": pos_val,        "required": f"< {max_pos} concurrent"},
                {"id": "rsi_bear_range", "label": "RSI Bear Range (10-bar)",  "pass": rsi_br_ok,         "value": rsi_br_val,     "required": f"10-bar RSI max < {rsi_br_max} (no hidden bullish momentum)"},
                {"id": "macd",           "label": "MACD(8,17,9) Histogram (info)", "pass": macd_ok,      "value": macd_val,       "required": "< 0 (informational — not a hard gate)"},
                {"id": "stochrsi",       "label": "StochRSI K (info only)",   "pass": stochrsi_k is not None and stochrsi_k > 60, "value": stochrsi_val, "required": "> 60 at entry (informational — not a hard gate)"},
            ]

            # Legacy flat list kept for backward compat
            signal_ready = short_signal_ready if bias == "short" else long_signal_ready
            bias_conditions = short_conditions if bias == "short" else long_conditions

            data = {
                "symbol":             symbol,
                "timestamp":          utc_now.isoformat(),
                "price":              price,
                "bias":               bias,
                "signal_ready":       signal_ready,
                "long_signal_ready":  long_signal_ready,
                "short_signal_ready": short_signal_ready,
                "long_conditions":    long_conditions,
                "short_conditions":   short_conditions,
                "conditions":         bias_conditions,
            }

            self._set_cached('conditions', data)
            self._json_response(200, data)
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _handle_force_trade(self):
        """
        POST /force_trade  body: {"side": "long"|"short"}
        Opens a paper trade at current market price, bypassing strategy conditions.
        Used to verify the full execution → DB → dashboard pipeline.
        Only works in paper mode.
        """
        bot = self.bot_instance
        if not bot:
            self._json_response(503, {"error": "Bot not ready"})
            return
        if bot.mode != "paper":
            self._json_response(400, {"error": "force_trade only allowed in paper mode"})
            return
        try:
            import pandas as pd
            from utils import calculate_position_size, generate_trade_id, TradeRecord, Position
            from datetime import datetime, timezone

            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            side = body.get("side", "long").lower()
            if side not in ("long", "short"):
                self._json_response(400, {"error": "side must be 'long' or 'short'"})
                return

            # Fetch live price + ATR
            df_5m = bot.exchange.fetch_ohlcv(bot.symbol, '5m', limit=20)
            if df_5m.empty:
                self._json_response(503, {"error": "Cannot fetch market data"})
                return
            df_5m = bot.strategy.compute_indicators(df_5m)
            bar = df_5m.iloc[-1]
            price = float(bar['close'])
            atr   = float(bar['atr']) if not pd.isna(bar['atr']) else 5.0

            # SL / TP using configured multipliers
            if side == 'long':
                sl_price  = round(price - bot.strategy.sl_atr_mult * atr, 2)
                tp1_price = round(price + bot.strategy.tp1_atr_mult * atr, 2)
            else:
                sl_price  = round(price + bot.strategy.short_sl_atr_mult * atr, 2)
                tp1_price = round(price - bot.strategy.short_tp1_atr_mult * atr, 2)

            balance = bot.exchange.get_balance()
            qty = calculate_position_size(balance, bot.risk_pct, price, sl_price)
            if qty <= 0:
                qty = 0.001  # minimum fallback so the test always works

            trade_id = generate_trade_id(side)
            now = datetime.now(timezone.utc)

            # Create in-memory position
            new_pos = Position(
                trade_id=trade_id,
                symbol=bot.symbol,
                side=side,
                entry_price=price,
                entry_time=now,
                quantity=qty,
                remaining_qty=qty,
                sl_price=sl_price,
                tp1_price=tp1_price,
                atr_at_entry=atr,
                highest_since_entry=price if side == 'long' else 0,
                lowest_since_entry=price if side == 'short' else 999999
            )
            bot.positions.append(new_pos)

            # Persist to DB
            record = TradeRecord(
                trade_id=trade_id,
                symbol=bot.symbol,
                side=side,
                entry_price=price,
                entry_time=now.isoformat(),
                quantity=qty,
                sl_price=sl_price,
                tp1_price=tp1_price,
                entry_reason="MANUAL force_trade test",
                remaining_qty=qty
            )
            bot.db.insert_trade(record)

            self._json_response(200, {
                "ok": True,
                "trade_id": trade_id,
                "side": side,
                "entry_price": price,
                "sl_price": sl_price,
                "tp1_price": tp1_price,
                "quantity": qty,
                "balance": balance,
            })
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _respond_trades(self):
        bot = self.bot_instance
        if not bot:
            self._json_response(503, {"error": "Bot not ready"})
            return
        try:
            trades = bot.db.get_all_trades()
            closed = [t for t in trades if t['status'] == 'closed']
            open_  = [t for t in trades if t['status'] in ('open', 'partial')]
            wins   = [t for t in closed if t['pnl'] > 0]
            losses = [t for t in closed if t['pnl'] <= 0]
            total_pnl   = round(sum(t['pnl'] for t in closed), 2)
            win_rate    = round(len(wins) / len(closed) * 100, 1) if closed else 0.0
            avg_win     = round(sum(t['pnl'] for t in wins)   / len(wins),   2) if wins   else 0.0
            avg_loss    = round(sum(t['pnl'] for t in losses) / len(losses), 2) if losses else 0.0

            # Return most recent 50 trades (newest first) for display
            recent = sorted(trades, key=lambda t: t['entry_time'], reverse=True)[:50]
            self._json_response(200, {
                "summary": {
                    "total_trades":  len(closed),
                    "open_trades":   len(open_),
                    "wins":          len(wins),
                    "losses":        len(losses),
                    "win_rate_pct":  win_rate,
                    "total_pnl":     total_pnl,
                    "avg_win":       avg_win,
                    "avg_loss":      avg_loss,
                },
                "trades": recent,
            })
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def log_message(self, format, *args):
        # Suppress default access logs (too noisy with health checks)
        pass


def start_health_server(port: int):
    """Run the health-check HTTP server in a background thread."""
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ── Main entry point ────────────────────────────────────────────────────────

def main():
    import tempfile
    
    # Build config from env vars
    config_dict = build_config_from_env()
    mode = os.getenv("BOT_MODE", "paper")
    port = int(os.getenv("PORT", 8000))
    
    # Validate API keys for live mode
    if mode == "live" and not config_dict["exchange"].get("api_key"):
        print("ERROR: EXCHANGE_API_KEY env var required for live mode!")
        sys.exit(1)
    
    # Write config to temp file (ConfigLoader reads from file)
    config_path = os.path.join(tempfile.gettempdir(), "bot_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Import after config is ready
    from utils import ConfigLoader, setup_logging
    from main import TradingBot
    
    config = ConfigLoader(config_path)
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("  GOLD SCALPER BOT — KOYEB DEPLOYMENT")
    logger.info(f"  Mode:     {mode.upper()}")
    logger.info(f"  Symbol:   {config['symbol']}")
    logger.info(f"  Exchange: {config.get('exchange', 'name')}")
    logger.info(f"  Testnet:  {config.get('exchange', 'testnet')}")
    logger.info(f"  Health:   http://0.0.0.0:{port}/health")
    logger.info("=" * 60)
    
    # Start health-check server
    health_server = start_health_server(port)
    logger.info(f"✅ Health-check server started on port {port}")
    
    # Create bot
    bot = TradingBot(config, mode=mode)
    HealthHandler.bot_instance = bot
    
    # Graceful shutdown
    def shutdown(sig, frame):
        logger.info(f"\n⏹ Shutdown signal received ({sig})...")
        bot.stop()
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Run bot
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Bot stopped.")
    finally:
        health_server.shutdown()
        logger.info("Health server stopped. Goodbye!")


if __name__ == "__main__":
    main()
