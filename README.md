# EUR/USD London Breakout Algo Bot

A production-grade algorithmic trading bot targeting prop firm challenges
(FTMO / E8) using a London Session Breakout strategy on EUR/USD.

---

## Strategy

**London Breakout** — captures the directional expansion that typically occurs
when the London session opens through the Asian consolidation range.

| Step | Detail |
|------|--------|
| Asian Range | Measures the high/low between 00:00–07:00 UTC |
| Valid Range | 10–50 pips — rejects choppy or already-extended sessions |
| Entry | Price breaks 2 pips above/below the Asian range at 07:00–10:00 UTC |
| RSI Filter | RSI(14) > 50 for BUY, < 50 for SELL (H1 timeframe) |
| Stop Loss | Opposite side of the Asian range + 2 pip buffer |
| Take Profit | 1:2 Risk-to-Reward (fixed) |
| Position Sizing | 1% fixed-fractional risk per trade |

---

## Risk Management (FTMO / E8 compliant)

| Rule | Setting |
|------|---------|
| Max daily loss | 4% of daily starting balance |
| Max total drawdown | 8% of starting balance |
| Max open positions | 3 |
| Max lot size | 5.0 |
| Friday close | All positions closed at 21:00 UTC |
| News filter | Blocks ±30/60 min around high-impact EUR/USD events |

---

## Project Structure

```
eurusd-algo-bot/
├── main.py              # Entry point + CLI (live / paper / backtest)
├── config.py            # Pydantic settings loaded from .env
├── bot/
│   ├── data_feed.py     # MT5 data access (candles, ticks, account)
│   ├── risk_manager.py  # Drawdown & daily loss enforcement
│   ├── strategy.py      # London Breakout signal generation
│   ├── executor.py      # MT5 order placement & management
│   ├── news_filter.py   # Forex Factory high-impact event filter
│   └── notifier.py      # Telegram alerts (non-blocking)
├── backtest/
│   ├── engine.py        # Vectorised backtesting on MT5 H1 data
│   └── report.py        # Interactive HTML report (Plotly)
├── tests/               # pytest test suite
├── logs/                # Rotating log files + backtest report
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- MetaTrader 5 terminal installed and logged in
- TA-Lib C library (see installation below)

### 2. Install TA-Lib (macOS)

```bash
brew install ta-lib
```

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your MT5 credentials and Telegram token
```

### 5. Run

```bash
# Paper mode (no real orders)
python main.py paper

# Live mode
python main.py live

# Backtest (requires MT5 with EURUSD H1 history 2021-2024)
python main.py backtest --output logs/backtest_report.html
```

---

## Docker

```bash
# Build
docker build -t eurusd-algo-bot .

# Run (paper mode, logs persisted)
docker run -d \
  --name eurusd-bot \
  --restart always \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  eurusd-algo-bot paper
```

> **Note:** MetaTrader5 Python package requires Windows or Wine on Linux.
> The Docker image is best suited for running the backtest or paper-mode
> simulation on a Windows host where MT5 is installed.

---

## Tests

```bash
pytest tests/ -v
```

---

## Configuration Reference

All settings live in `.env` and are validated by Pydantic on startup.

| Key | Default | Description |
|-----|---------|-------------|
| `MT5_LOGIN` | — | MT5 account number |
| `MT5_PASSWORD` | — | MT5 account password |
| `MT5_SERVER` | — | Broker server name |
| `TELEGRAM_BOT_TOKEN` | — | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | — | Your chat or group ID |
| `SYMBOL` | `EURUSD` | Trading instrument |
| `RISK_PER_TRADE` | `0.01` | Fraction of equity risked per trade (max 0.02) |
| `MAX_DAILY_LOSS` | `0.04` | Daily drawdown limit (max 0.05) |
| `MAX_TOTAL_DRAWDOWN` | `0.08` | Total drawdown limit |
| `MAX_LOT_SIZE` | `5.0` | Hard lot cap |

---

## Telegram Alerts

The bot sends the following notifications:

- **Trade executed** — direction, entry, SL, TP, lot, RSI, confidence
- **Risk alert** — when approaching daily/total drawdown limits
- **Daily summary** — EOD trade count, win rate, net P&L
- **Error alert** — any unhandled exception

---

## Disclaimer

This software is for educational and research purposes. Algorithmic trading
involves substantial risk of loss. Past backtest performance does not guarantee
future results. Always test thoroughly in paper mode before risking real capital.
