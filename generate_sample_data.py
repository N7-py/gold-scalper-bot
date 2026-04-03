"""
generate_sample_data.py — Generate synthetic XAUUSDT 5M candle data for backtest testing.

This creates realistic-looking gold price data with trends, pullbacks, and volatility
so you can test the bot without needing real exchange data downloads.

Usage:
  python generate_sample_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os


def generate_gold_data(
    start_date: str = '2024-01-01',
    days: int = 180,
    start_price: float = 2050.0,
    volatility: float = 0.0008,
    trend_strength: float = 0.00002
) -> pd.DataFrame:
    """
    Generate synthetic 5M OHLCV data resembling XAUUSDT price action.
    
    Creates realistic patterns including:
    - Trend phases (up/down)
    - Pullbacks to moving averages
    - Session-based volatility (London/NY more active)
    - Occasional spikes
    """
    np.random.seed(42)
    
    # 5M bars per day = 288
    bars_per_day = 288
    total_bars = days * bars_per_day
    
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    price = start_price
    trend = trend_strength
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    
    for i in range(total_bars):
        dt = start_dt + timedelta(minutes=5 * i)
        hour = dt.hour
        
        # Session-based volatility multiplier
        if 8 <= hour <= 12:       # London
            vol_mult = 1.5
        elif 13 <= hour <= 17:    # NY overlap
            vol_mult = 1.8
        elif 0 <= hour <= 7:      # Asia
            vol_mult = 0.7
        else:
            vol_mult = 0.9
        
        # Change trend direction periodically (every ~15-30 days)
        if i % (bars_per_day * np.random.randint(15, 30)) == 0 and i > 0:
            trend = -trend * np.random.uniform(0.5, 1.5)
        
        # Generate price movement
        noise = np.random.normal(0, volatility * vol_mult)
        change = trend + noise
        
        # Occasional spike (news event simulation)
        if np.random.random() < 0.001:
            change += np.random.choice([-1, 1]) * volatility * 10
        
        open_price = price
        close_price = price * (1 + change)
        
        # Generate high/low
        bar_range = abs(change) + volatility * vol_mult * 0.5
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, bar_range * 0.3)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, bar_range * 0.3)))
        
        # Volume (correlated with volatility)
        base_vol = 100 + abs(change) * 50000
        volume = base_vol * vol_mult * np.random.uniform(0.5, 2.0)
        
        timestamps.append(dt)
        opens.append(round(open_price, 2))
        highs.append(round(high_price, 2))
        lows.append(round(low_price, 2))
        closes.append(round(close_price, 2))
        volumes.append(round(volume, 2))
        
        price = close_price
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    return df


def main():
    print("Generating synthetic XAUUSDT 5M data...")
    
    # Generate 6 months of data
    df = generate_gold_data(
        start_date='2024-01-01',
        days=180,
        start_price=2050.0
    )
    
    # Save
    os.makedirs('data', exist_ok=True)
    output_path = 'data/XAUUSDT_5m.csv'
    df.to_csv(output_path, index=False)
    
    print(f"[OK] Generated {len(df)} bars ({df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']})")
    print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"   Saved to: {output_path}")
    
    # Also create 1H resampled
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_1h = df.set_index('timestamp').resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    output_1h = 'data/XAUUSDT_1h.csv'
    df_1h.to_csv(output_1h, index=False)
    print(f"   1H resampled: {len(df_1h)} bars -> {output_1h}")


if __name__ == '__main__':
    main()
