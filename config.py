from __future__ import annotations

from typing import Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # MT5 connection — Optional so empty strings in .env are accepted
    mt5_login: Optional[int] = None
    mt5_password: str = ""
    mt5_server: str = ""

    # Telegram — Optional so empty strings in .env are accepted
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Trading parameters
    symbol: str = "EURUSD"
    paper_starting_balance: float = 10_000.0
    risk_per_trade: float = 0.01
    max_daily_loss: float = 0.04
    max_total_drawdown: float = 0.07
    max_lot_size: float = 5.0
    max_open_positions: int = 1
    order_magic_id: int = 202400  # unique EA identifier sent with every order

    # Strategy parameters
    asian_session_start_utc: int = 0    # 00:00 UTC
    asian_session_end_utc: int = 7      # 07:00 UTC
    london_session_start_utc: int = 7   # 07:00 UTC
    london_session_end_utc: int = 10    # 10:00 UTC

    min_range_pips: float = 10.0
    max_range_pips: float = 50.0
    breakout_buffer_pips: float = 2.0
    rr_ratio: float = 1.5               # 1:1.5 risk/reward
    rsi_period: int = 14

    # News filter
    news_filter_before_minutes: int = 30
    news_filter_after_minutes: int = 60

    # Execution
    max_slippage_pips: float = 3.0
    order_retry_attempts: int = 3
    order_retry_delay_seconds: float = 1.0
    order_filling_mode: int = 0  # 0=FOK, 1=IOC, 2=RETURN

    # Risk checks
    friday_close_hour_utc: int = 21

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("mt5_login", mode="before")
    @classmethod
    def coerce_mt5_login(cls, v: object) -> Optional[int]:
        """Accept empty string from .env as None (no login configured)."""
        if v == "" or v is None:
            return None
        return int(str(v))

    @field_validator("risk_per_trade")
    @classmethod
    def validate_risk_per_trade(cls, v: float) -> float:
        if v > 0.02:
            raise ValueError(
                f"risk_per_trade={v} exceeds maximum allowed 0.02 (2%)"
            )
        if v <= 0:
            raise ValueError("risk_per_trade must be positive")
        return v

    @field_validator("max_daily_loss")
    @classmethod
    def validate_max_daily_loss(cls, v: float) -> float:
        if v > 0.05:
            raise ValueError(
                f"max_daily_loss={v} exceeds maximum allowed 0.05 (5%)"
            )
        if v <= 0:
            raise ValueError("max_daily_loss must be positive")
        return v

    @model_validator(mode="after")
    def validate_drawdown_greater_than_daily(self) -> "Settings":
        if self.max_total_drawdown <= self.max_daily_loss:
            raise ValueError(
                "max_total_drawdown must be greater than max_daily_loss"
            )
        return self


settings = Settings()
