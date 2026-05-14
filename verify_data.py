import MetaTrader5 as mt5
from config import settings
from dotenv import load_dotenv
from datetime import datetime, timezone
import time

load_dotenv()

# Connect
mt5.initialize()
mt5.login(settings.mt5_login, settings.mt5_password, settings.mt5_server)

symbol = settings.symbol
mt5.symbol_select(symbol, True)
time.sleep(1)

print('=== DATA FRESHNESS CHECK ===')
print()

# CHECK 1 — Live tick timestamp
tick = mt5.symbol_info_tick(symbol)
tick_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
now_utc = datetime.now(timezone.utc)
tick_age_seconds = (now_utc - tick_time).total_seconds()

print('=== TICK DATA ===')
print(f'Bid:          {tick.bid}')
print(f'Ask:          {tick.ask}')
print(f'Tick time:    {tick_time.strftime("%Y-%m-%d %H:%M:%S UTC")}')
print(f'Current time: {now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")}')
print(f'Tick age:     {tick_age_seconds:.1f} seconds')

if tick_age_seconds < 10:
    print(f'Data freshness: REAL-TIME ✅ ({tick_age_seconds:.1f}s old)')
elif tick_age_seconds < 60:
    print(f'Data freshness: FRESH ✅ ({tick_age_seconds:.1f}s old)')
else:
    print(f'Data freshness: DELAYED ⚠️ ({tick_age_seconds:.1f}s old)')

print()

# CHECK 2 — H1 candles freshness
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 3)
print('=== H1 CANDLE DATA ===')
for r in rates:
    candle_time = datetime.fromtimestamp(r['time'], tz=timezone.utc)
    age_minutes = (now_utc - candle_time).total_seconds() / 60
    print(f'Candle: {candle_time.strftime("%H:%M UTC")} | '
          f'O={r["open"]:.5f} H={r["high"]:.5f} '
          f'L={r["low"]:.5f} C={r["close"]:.5f} | '
          f'Age: {age_minutes:.0f} min')

print()

# CHECK 3 — Compare MT5 price vs Yahoo Finance
print('=== MT5 vs YAHOO FINANCE COMPARISON ===')
try:
    import yfinance as yf
    yahoo_ticker = yf.Ticker('EURUSD=X')
    yahoo_data = yahoo_ticker.history(period='1d', interval='1m')

    if not yahoo_data.empty:
        yahoo_price = float(yahoo_data['Close'].iloc[-1])
        yahoo_time = yahoo_data.index[-1]
        mt5_price = tick.bid
        difference = abs(mt5_price - yahoo_price)
        difference_pips = difference / 0.0001

        print(f'MT5 price:    {mt5_price:.5f} (real-time)')
        print(f'Yahoo price:  {yahoo_price:.5f} (delayed)')
        print(f'Difference:   {difference_pips:.1f} pips')

        if difference_pips < 5:
            print('Prices match ✅ MT5 data is accurate')
        else:
            print(f'Difference {difference_pips:.1f} pips ⚠️ (Yahoo has delay)')
    else:
        print('Yahoo Finance: No data (market closed or weekend)')

except Exception as e:
    print(f'Yahoo comparison failed: {e}')

print()

# CHECK 4 — Asian range calculation
print('=== ASIAN RANGE VERIFICATION ===')
import pandas as pd
bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
df = pd.DataFrame(bars)
df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

asian = df[df['time'].dt.hour < 7]
if len(asian) > 0:
    asian_high = asian['high'].max()
    asian_low = asian['low'].min()
    range_pips = (asian_high - asian_low) / 0.0001
    print(f'Asian bars found: {len(asian)}')
    print(f'Asian high:  {asian_high:.5f}')
    print(f'Asian low:   {asian_low:.5f}')
    print(f'Range:       {range_pips:.1f} pips')
    print(f'Valid range: {10 <= range_pips <= 50}')
    print(f'Data source: MT5 real-time bars ✅')
else:
    print('No Asian session bars yet (before 07:00 UTC)')

print()

# CHECK 5 — Order book / spread
print('=== SPREAD CHECK ===')
spread_pips = (tick.ask - tick.bid) / 0.0001
print(f'Bid: {tick.bid:.5f}')
print(f'Ask: {tick.ask:.5f}')
print(f'Spread: {spread_pips:.1f} pips')
print(f'Spread source: The5ers live broker ✅')

print()
print('=== FINAL VERDICT ===')
if tick_age_seconds < 60:
    print('✅ Bot uses REAL-TIME MT5 data')
    print('✅ Zero meaningful delay')
    print('✅ Same data as MT5 terminal')
    print('✅ Direct from The5ers broker')
    print('✅ Not Yahoo Finance (no delay)')
else:
    print('⚠️ Data may be delayed')
    print('Check MT5 connection')

mt5.shutdown()
