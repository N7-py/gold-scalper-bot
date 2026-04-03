"""
app_gui.py - Gold Momentum Multi-TF Scalper Bot — Desktop GUI
Premium CustomTkinter interface with online/offline status,
mode selection, live log console, and trade statistics.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue
import logging
import sys
import os
import json
import time
from datetime import datetime, timezone
from io import StringIO

# ── Ensure imports work when bundled as exe ──
def get_base_path():
    """Get the base path for resources (works for both dev and PyInstaller)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()
sys.path.insert(0, BASE_PATH)
os.chdir(BASE_PATH)

# ── Theme & Colors ──
COLORS = {
    "bg_dark":       "#0D1117",
    "bg_card":       "#161B22",
    "bg_card_alt":   "#1C2333",
    "bg_input":      "#21262D",
    "border":        "#30363D",
    "text_primary":  "#E6EDF3",
    "text_secondary":"#8B949E",
    "text_muted":    "#484F58",
    "accent_gold":   "#F0B90B",
    "accent_green":  "#2EA043",
    "accent_green_bright": "#3FB950",
    "accent_red":    "#F85149",
    "accent_blue":   "#58A6FF",
    "accent_purple": "#BC8CFF",
    "online_glow":   "#00FF6A",
    "offline_glow":  "#FF4444",
}


class QueueHandler(logging.Handler):
    """Thread-safe logging handler that sends records to a queue."""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            pass


class GoldScalperApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # ── Window Setup ──
        self.title("Gold Momentum Scalper")
        self.geometry("1100x750")
        self.minsize(900, 650)
        self.configure(fg_color=COLORS["bg_dark"])

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # ── State ──
        self.bot_thread = None
        self.bot_running = False
        self.bot_status = "offline"  # "online", "offline", "starting"
        self.log_queue = queue.Queue()
        self.config_path = os.path.join(BASE_PATH, "config.json")
        self.trade_count = 0
        self.session_pnl = 0.0
        self.start_time = None
        self.mode_var = ctk.StringVar(value="backtest")
        self.data_path_var = ctk.StringVar(value="")

        # ── Build UI ──
        self._build_header()
        self._build_main_area()
        self._build_footer()

        # ── Poll log queue ──
        self._poll_logs()

        # ── Handle close ──
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────
    #  HEADER
    # ─────────────────────────────────────────────
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=0, height=70)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        # Left: Title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", padx=20, pady=10)

        ctk.CTkLabel(
            title_frame, text="GOLD MOMENTUM SCALPER",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=COLORS["accent_gold"]
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_frame, text="Multi-TF Strategy  |  XAUUSDT",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        ).pack(anchor="w")

        # Right: Status indicator
        status_frame = ctk.CTkFrame(header, fg_color="transparent")
        status_frame.pack(side="right", padx=20, pady=10)

        self.status_dot = ctk.CTkLabel(
            status_frame, text="\u2B24", font=ctk.CTkFont(size=16),
            text_color=COLORS["offline_glow"]
        )
        self.status_dot.pack(side="left", padx=(0, 8))

        self.status_label = ctk.CTkLabel(
            status_frame, text="OFFLINE",
            font=ctk.CTkFont(family="Consolas", size=16, weight="bold"),
            text_color=COLORS["accent_red"]
        )
        self.status_label.pack(side="left")

    # ─────────────────────────────────────────────
    #  MAIN AREA
    # ─────────────────────────────────────────────
    def _build_main_area(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=(12, 0))

        # Left panel: Controls + Stats
        left = ctk.CTkFrame(main, fg_color="transparent", width=320)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        self._build_controls(left)
        self._build_stats(left)

        # Right panel: Console
        right = ctk.CTkFrame(main, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)

        self._build_console(right)

    def _build_controls(self, parent):
        card = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=12,
                            border_width=1, border_color=COLORS["border"])
        card.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            card, text="CONTROLS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_muted"]
        ).pack(anchor="w", padx=16, pady=(14, 8))

        # Separator
        sep = ctk.CTkFrame(card, fg_color=COLORS["border"], height=1)
        sep.pack(fill="x", padx=16, pady=(0, 12))

        # Mode selection
        ctk.CTkLabel(
            card, text="Mode",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"]
        ).pack(anchor="w", padx=16, pady=(0, 4))

        mode_frame = ctk.CTkFrame(card, fg_color="transparent")
        mode_frame.pack(fill="x", padx=16, pady=(0, 12))

        modes = [("Backtest", "backtest"), ("Paper", "paper"), ("Live", "live")]
        for i, (text, val) in enumerate(modes):
            btn = ctk.CTkRadioButton(
                mode_frame, text=text, variable=self.mode_var, value=val,
                font=ctk.CTkFont(size=12),
                text_color=COLORS["text_primary"],
                fg_color=COLORS["accent_gold"],
                hover_color=COLORS["accent_gold"],
                border_color=COLORS["border"],
                command=self._on_mode_change
            )
            btn.pack(anchor="w", pady=2)

        # Data file (for backtest)
        self.data_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.data_frame.pack(fill="x", padx=16, pady=(0, 12))

        ctk.CTkLabel(
            self.data_frame, text="Data File",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"]
        ).pack(anchor="w", pady=(0, 4))

        file_row = ctk.CTkFrame(self.data_frame, fg_color="transparent")
        file_row.pack(fill="x")

        self.data_entry = ctk.CTkEntry(
            file_row, textvariable=self.data_path_var,
            font=ctk.CTkFont(size=11),
            fg_color=COLORS["bg_input"],
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            placeholder_text="Select CSV/Parquet...",
            height=32
        )
        self.data_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        ctk.CTkButton(
            file_row, text="...", width=36, height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["border"],
            border_width=1, border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            command=self._browse_data
        ).pack(side="right")

        # Start / Stop button
        self.start_btn = ctk.CTkButton(
            card, text="START BOT",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLORS["accent_green"],
            hover_color=COLORS["accent_green_bright"],
            text_color="#FFFFFF",
            height=44,
            corner_radius=8,
            command=self._toggle_bot
        )
        self.start_btn.pack(fill="x", padx=16, pady=(4, 16))

    def _build_stats(self, parent):
        card = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=12,
                            border_width=1, border_color=COLORS["border"])
        card.pack(fill="x", expand=True, pady=(0, 0))

        ctk.CTkLabel(
            card, text="SESSION STATS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_muted"]
        ).pack(anchor="w", padx=16, pady=(14, 8))

        sep = ctk.CTkFrame(card, fg_color=COLORS["border"], height=1)
        sep.pack(fill="x", padx=16, pady=(0, 12))

        # Stats grid
        stats_data = [
            ("Status",   "status_val",   "OFFLINE",  COLORS["accent_red"]),
            ("Mode",     "mode_val",     "Backtest", COLORS["text_primary"]),
            ("Runtime",  "runtime_val",  "00:00:00", COLORS["text_primary"]),
            ("Trades",   "trades_val",   "0",        COLORS["accent_blue"]),
            ("PnL",      "pnl_val",      "$0.00",    COLORS["text_primary"]),
            ("Win Rate", "winrate_val",  "—",        COLORS["text_primary"]),
        ]

        for label_text, attr_name, default_val, color in stats_data:
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=3)

            ctk.CTkLabel(
                row, text=label_text,
                font=ctk.CTkFont(size=12),
                text_color=COLORS["text_secondary"]
            ).pack(side="left")

            val_label = ctk.CTkLabel(
                row, text=default_val,
                font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
                text_color=color
            )
            val_label.pack(side="right")
            setattr(self, attr_name, val_label)

        # Spacer at bottom
        ctk.CTkFrame(card, fg_color="transparent", height=14).pack()

    def _build_console(self, parent):
        card = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=12,
                            border_width=1, border_color=COLORS["border"])
        card.pack(fill="both", expand=True)

        # Console header
        console_header = ctk.CTkFrame(card, fg_color="transparent")
        console_header.pack(fill="x", padx=16, pady=(14, 0))

        ctk.CTkLabel(
            console_header, text="LIVE CONSOLE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_muted"]
        ).pack(side="left")

        ctk.CTkButton(
            console_header, text="Clear", width=60, height=24,
            font=ctk.CTkFont(size=10),
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["border"],
            text_color=COLORS["text_secondary"],
            corner_radius=4,
            command=self._clear_console
        ).pack(side="right")

        sep = ctk.CTkFrame(card, fg_color=COLORS["border"], height=1)
        sep.pack(fill="x", padx=16, pady=(10, 8))

        # Console text area
        self.console = ctk.CTkTextbox(
            card,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=COLORS["bg_dark"],
            text_color=COLORS["text_primary"],
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
            wrap="word",
            state="disabled"
        )
        self.console.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Configure colored tags
        self.console._textbox.tag_config("info", foreground=COLORS["text_primary"])
        self.console._textbox.tag_config("warning", foreground=COLORS["accent_gold"])
        self.console._textbox.tag_config("error", foreground=COLORS["accent_red"])
        self.console._textbox.tag_config("success", foreground=COLORS["accent_green_bright"])
        self.console._textbox.tag_config("signal", foreground=COLORS["accent_blue"])
        self.console._textbox.tag_config("header", foreground=COLORS["accent_gold"])
        self.console._textbox.tag_config("muted", foreground=COLORS["text_muted"])

    # ─────────────────────────────────────────────
    #  FOOTER
    # ─────────────────────────────────────────────
    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=0, height=32)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        ctk.CTkLabel(
            footer, text="Gold Momentum Scalper v1.0  |  XAUUSDT  |  Binance",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_muted"]
        ).pack(side="left", padx=16)

        self.clock_label = ctk.CTkLabel(
            footer, text="",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color=COLORS["text_muted"]
        )
        self.clock_label.pack(side="right", padx=16)
        self._update_clock()

    # ─────────────────────────────────────────────
    #  EVENT HANDLERS
    # ─────────────────────────────────────────────
    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "backtest":
            self.data_frame.pack(fill="x", padx=16, pady=(0, 12))
        else:
            self.data_frame.pack_forget()
        self.mode_val.configure(text=mode.capitalize())

    def _browse_data(self):
        path = filedialog.askopenfilename(
            title="Select historical data file",
            filetypes=[("CSV files", "*.csv"), ("Parquet files", "*.parquet"), ("All files", "*.*")],
            initialdir=os.path.join(BASE_PATH, "data")
        )
        if path:
            self.data_path_var.set(path)

    def _toggle_bot(self):
        if self.bot_running:
            self._stop_bot()
        else:
            self._start_bot()

    def _start_bot(self):
        mode = self.mode_var.get()

        # Validate
        if mode == "backtest":
            data_path = self.data_path_var.get().strip()
            if not data_path or not os.path.exists(data_path):
                messagebox.showerror("Error", "Please select a valid data file for backtest mode.")
                return

        if mode == "live":
            try:
                with open(self.config_path, 'r') as f:
                    cfg = json.load(f)
                if cfg.get('exchange', {}).get('api_key', '') == 'YOUR_API_KEY_HERE':
                    resp = messagebox.askyesno(
                        "Warning",
                        "API keys are not configured in config.json.\n"
                        "The bot will fail to connect.\n\nContinue anyway?"
                    )
                    if not resp:
                        return
            except Exception:
                pass

        # Update UI
        self._set_status("starting")
        self.start_btn.configure(
            text="STOP BOT",
            fg_color=COLORS["accent_red"],
            hover_color="#D73A3A"
        )
        self.bot_running = True
        self.start_time = time.time()
        self.trade_count = 0
        self.session_pnl = 0.0
        self._update_runtime()

        self._log_to_console("=" * 58, "header")
        self._log_to_console("  GOLD MOMENTUM MULTI-TF SCALPER BOT", "header")
        self._log_to_console(f"  Mode: {mode.upper()}", "header")
        self._log_to_console("=" * 58, "header")

        # Start bot in background thread
        self.bot_thread = threading.Thread(target=self._run_bot_thread, args=(mode,), daemon=True)
        self.bot_thread.start()

    def _stop_bot(self):
        self.bot_running = False
        self._set_status("offline")
        self.start_btn.configure(
            text="START BOT",
            fg_color=COLORS["accent_green"],
            hover_color=COLORS["accent_green_bright"]
        )
        self._log_to_console("Bot stopped by user.", "warning")

    def _run_bot_thread(self, mode: str):
        """Run the trading bot in a background thread."""
        try:
            # Set up logging to capture into our queue
            logger = logging.getLogger('GoldScalper')
            logger.handlers.clear()
            logger.setLevel(logging.INFO)

            queue_handler = QueueHandler(self.log_queue)
            queue_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            ))
            logger.addHandler(queue_handler)

            # Also add file handler
            fh = logging.FileHandler(os.path.join(BASE_PATH, "bot.log"), encoding='utf-8')
            fh.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(fh)

            # Import bot modules
            from utils import ConfigLoader
            config = ConfigLoader(self.config_path)

            self._set_status("online")

            if mode == "backtest":
                self._run_backtest(config, logger)
            elif mode == "paper":
                self._run_paper(config, logger)
            elif mode == "live":
                self._run_live(config, logger)

        except Exception as e:
            self.log_queue.put(f"ERROR | Fatal: {e}")
            import traceback
            self.log_queue.put(traceback.format_exc())
        finally:
            if self.bot_running:
                self.bot_running = False
                self.after(0, lambda: self._set_status("offline"))
                self.after(0, lambda: self.start_btn.configure(
                    text="START BOT",
                    fg_color=COLORS["accent_green"],
                    hover_color=COLORS["accent_green_bright"]
                ))

    def _run_backtest(self, config, logger):
        """Run backtest mode."""
        from backtester import Backtester, DataLoader

        data_path = self.data_path_var.get().strip()
        logger.info(f"Loading data from: {data_path}")

        df_5m = DataLoader.load(data_path, '5m')
        logger.info(f"Loaded {len(df_5m)} bars")

        backtester = Backtester(config, logger)
        result = backtester.run(df_5m)

        # Update stats
        self.trade_count = result.total_trades
        self.session_pnl = result.net_profit
        self.after(0, lambda: self._update_stats_display(result))

        # Plot
        try:
            backtester.plot_equity_curve(result, os.path.join(BASE_PATH, "equity_curve.png"))
        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")

        # Export
        backtester.export_trades_csv(result, os.path.join(BASE_PATH, "backtest_trades.csv"))

        logger.info("Backtest complete!")

    def _run_paper(self, config, logger):
        """Run paper trading mode."""
        import asyncio
        from main import TradingBot

        bot = TradingBot(config, mode='paper')

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(bot.start())
        except Exception as e:
            if self.bot_running:
                logger.error(f"Paper trading error: {e}")
        finally:
            loop.close()

    def _run_live(self, config, logger):
        """Run live trading mode."""
        import asyncio
        from main import TradingBot

        bot = TradingBot(config, mode='live')

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(bot.start())
        except Exception as e:
            if self.bot_running:
                logger.error(f"Live trading error: {e}")
        finally:
            loop.close()

    # ─────────────────────────────────────────────
    #  STATUS & UI UPDATES
    # ─────────────────────────────────────────────
    def _set_status(self, status: str):
        """Update the status indicator (thread-safe via after)."""
        def _update():
            self.bot_status = status
            if status == "online":
                self.status_dot.configure(text_color=COLORS["online_glow"])
                self.status_label.configure(text="ONLINE", text_color=COLORS["accent_green_bright"])
                self.status_val.configure(text="ONLINE", text_color=COLORS["accent_green_bright"])
            elif status == "starting":
                self.status_dot.configure(text_color=COLORS["accent_gold"])
                self.status_label.configure(text="STARTING...", text_color=COLORS["accent_gold"])
                self.status_val.configure(text="STARTING", text_color=COLORS["accent_gold"])
            else:
                self.status_dot.configure(text_color=COLORS["offline_glow"])
                self.status_label.configure(text="OFFLINE", text_color=COLORS["accent_red"])
                self.status_val.configure(text="OFFLINE", text_color=COLORS["accent_red"])

        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.after(0, _update)

    def _update_stats_display(self, result=None):
        """Update the stats panel."""
        self.trades_val.configure(text=str(self.trade_count))
        pnl_color = COLORS["accent_green_bright"] if self.session_pnl >= 0 else COLORS["accent_red"]
        self.pnl_val.configure(text=f"${self.session_pnl:+,.2f}", text_color=pnl_color)

        if result and result.total_trades > 0:
            self.winrate_val.configure(text=f"{result.win_rate:.1f}%")

    def _update_runtime(self):
        """Update runtime counter every second."""
        if self.bot_running and self.start_time:
            elapsed = int(time.time() - self.start_time)
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            self.runtime_val.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
            self.after(1000, self._update_runtime)
        else:
            pass  # Stop updating

    def _update_clock(self):
        """Update footer clock."""
        now = datetime.now()
        utc = datetime.now(timezone.utc)
        self.clock_label.configure(
            text=f"Local: {now.strftime('%H:%M:%S')}  |  UTC: {utc.strftime('%H:%M:%S')}"
        )
        self.after(1000, self._update_clock)

    # ─────────────────────────────────────────────
    #  CONSOLE
    # ─────────────────────────────────────────────
    def _log_to_console(self, message: str, tag: str = "info"):
        """Write a message to the console textbox."""
        self.console.configure(state="normal")
        self.console.insert("end", message + "\n", tag)
        self.console.see("end")
        self.console.configure(state="disabled")

    def _clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def _poll_logs(self):
        """Poll the log queue and display messages in the console."""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                # Color-code based on content
                tag = "info"
                msg_upper = msg.upper()
                if "ERROR" in msg_upper or "FATAL" in msg_upper:
                    tag = "error"
                elif "WARNING" in msg_upper or "EMERGENCY" in msg_upper:
                    tag = "warning"
                elif "SIGNAL" in msg_upper or "LONG" in msg_upper or "SHORT" in msg_upper:
                    tag = "signal"
                elif "ENTRY" in msg_upper or "CLOSED" in msg_upper or "TP1" in msg_upper:
                    tag = "success"
                elif "====" in msg or "----" in msg:
                    tag = "header"

                # Track trades from log messages
                if "ENTRY" in msg_upper:
                    self.trade_count += 1
                    self.after(0, lambda: self.trades_val.configure(text=str(self.trade_count)))

                self._log_to_console(msg, tag)
        except queue.Empty:
            pass

        self.after(100, self._poll_logs)

    def _on_close(self):
        """Handle window close."""
        if self.bot_running:
            resp = messagebox.askyesno(
                "Confirm Exit",
                "The bot is still running.\nStop the bot and exit?"
            )
            if not resp:
                return
            self.bot_running = False

        self.destroy()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    app = GoldScalperApp()
    app.mainloop()


if __name__ == "__main__":
    main()
