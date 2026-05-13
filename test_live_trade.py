import time

import MetaTrader5 as mt5

from config import settings
from dotenv import load_dotenv

load_dotenv()

# ── 1. CONNECT TO MT5 ────────────────────────────────────────────────────────
mt5.initialize()
result = mt5.login(
    settings.mt5_login,
    settings.mt5_password,
    settings.mt5_server,
)
print("MT5 Login:", result)

info = mt5.account_info()
print("Account:", info.login)
print("Balance:", info.balance)
print("Server:", info.server)

# ── 2. GET LIVE PRICE ────────────────────────────────────────────────────────
symbol = settings.symbol
mt5.symbol_select(symbol, True)
time.sleep(1)

tick = mt5.symbol_info_tick(symbol)
print()
print("=== LIVE PRICE ===")
print("Symbol:", symbol)
print("Bid:", tick.bid)
print("Ask:", tick.ask)
print("Spread:", round((tick.ask - tick.bid) / 0.0001, 1), "pips")

# ── 3. CALCULATE LOT SIZE (minimum safe) ────────────────────────────────────
balance = info.balance
risk_amount = balance * 0.001  # 0.1% for test only
sl_pips = 20
pip_value = 10.0
lot_size = round(risk_amount / (sl_pips * pip_value), 2)
lot_size = max(0.01, lot_size)
print()
print("=== LOT SIZE ===")
print("Risk amount:", risk_amount)
print("Lot size:", lot_size)

# ── 4. PLACE TEST BUY ORDER ──────────────────────────────────────────────────
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot_size,
    "type": mt5.ORDER_TYPE_BUY,
    "price": tick.ask,
    "sl": round(tick.ask - 0.0020, 5),
    "tp": round(tick.ask + 0.0040, 5),
    "deviation": 20,
    "magic": settings.order_magic_id,
    "comment": "Bot test trade",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": settings.order_filling_mode,
}

print()
print("=== PLACING TEST TRADE ===")
print("Direction: BUY")
print("Volume:", lot_size, "lots")
print("Entry:", tick.ask)
print("SL:", request["sl"])
print("TP:", request["tp"])

result = mt5.order_send(request)
print()
print("Return code:", result.retcode)
print("Comment:", result.comment)

if result.retcode == 10009:
    print("TRADE PLACED ✅")
    print("Ticket:", result.order)
    ticket = result.order

    print()
    print("Waiting 3 seconds...")
    time.sleep(3)

    positions = mt5.positions_get(symbol=symbol)
    print()
    print("=== POSITION STATUS ===")
    if positions:
        pos = positions[0]
        print("Ticket:", pos.ticket)
        print("Type:", "BUY" if pos.type == 0 else "SELL")
        print("Volume:", pos.volume)
        print("Open price:", pos.price_open)
        current = mt5.symbol_info_tick(symbol)
        unrealized = (current.bid - pos.price_open) / 0.0001
        print("Current price:", current.bid)
        print("Unrealized pips:", round(unrealized, 1))
        print("Unrealized P&L: $", round(pos.profit, 2))

        print()
        print("=== CLOSING TEST TRADE ===")
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_SELL,
            "position": pos.ticket,
            "price": current.bid,
            "deviation": 20,
            "magic": settings.order_magic_id,
            "comment": "Close bot test",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": settings.order_filling_mode,
        }

        close_result = mt5.order_send(close_request)
        print("Close code:", close_result.retcode)

        if close_result.retcode == 10009:
            print("TRADE CLOSED ✅")
            print()
            print("=== TEST COMPLETE ===")
            print("✅ MT5 connection working")
            print("✅ Symbol feed working")
            print("✅ Order placement working")
            print("✅ Position monitoring working")
            print("✅ Order closing working")
            print("✅ Bot ready for live trading!")
        else:
            print("Close failed ❌:", close_result.comment)
    else:
        print("Position not found ❌")

else:
    print("TRADE FAILED ❌")
    print("Error:", result.retcode, result.comment)

    errors = {
        10004: "Requote",
        10006: "Request rejected",
        10007: "Request cancelled",
        10010: "Only part of request completed",
        10011: "Request processing error",
        10012: "Request cancelled by timeout",
        10013: "Invalid request",
        10014: "Invalid volume",
        10015: "Invalid price",
        10016: "Invalid SL/TP",
        10017: "Trade disabled",
        10018: "Market closed",
        10019: "Insufficient funds",
        10024: "Too many requests",
        10030: "Unsupported filling mode",
    }

    if result.retcode in errors:
        print("Reason:", errors[result.retcode])

# ── 5. FINAL SUMMARY ─────────────────────────────────────────────────────────
print()
print("=== ACCOUNT FINAL STATE ===")
info = mt5.account_info()
print("Balance:", info.balance)
print("Equity:", info.equity)
print("Profit:", info.profit)

mt5.shutdown()
