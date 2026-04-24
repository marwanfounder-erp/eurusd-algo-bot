FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────
# No Wine or MT5 needed — paper mode uses yfinance.
# libgomp1 is required by numpy/scipy on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ─────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────
COPY . .

# logs/ directory — Railway mounts a persistent volume here via their dashboard
RUN mkdir -p /app/logs

# ── Runtime ───────────────────────────────────────────────────────────────
# Single CMD form — no ENTRYPOINT so Railway can override the command cleanly.
CMD ["python", "main.py", "--paper"]

# ── Labels ────────────────────────────────────────────────────────────────
LABEL maintainer="eurusd-algo-bot" \
      version="1.1.0" \
      description="EUR/USD London Breakout algo bot — paper mode on Railway/Linux, live via MT5 on Windows"
