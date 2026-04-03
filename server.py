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
    config["strategy"].setdefault("pullback_atr_mult", 1.0)
    config["strategy"].setdefault("sl_atr_mult", 1.5)
    config["strategy"].setdefault("tp1_atr_mult", 2.0)
    config["strategy"].setdefault("trail_atr_mult", 1.0)
    config["strategy"].setdefault("tp1_close_pct", 0.5)
    
    # Session hours
    config.setdefault("session_hours_utc", {"london": [8, 12], "new_york": [13, 17]})
    
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
    
    def do_GET(self):
        if self.path == "/" or self.path == "/health":
            self._respond_health()
        elif self.path == "/status":
            self._respond_status()
        else:
            self.send_error(404)
    
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
