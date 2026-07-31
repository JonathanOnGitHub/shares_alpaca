"""Forward paper-trading for 200-day MA timing strategy.
Run daily via cron.

Tests the Burns & Holland market-timing approach:
  SPY > 200-day MA → stay invested
  SPY < 200-day MA → move to cash

Usage:
  python -m src.trading.ma_timing_forward
"""
import argparse
import logging

from src.data.alpaca_client import AlpacaClient
from src.trading.ma_timing import MATiming

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def get_signal(client: AlpacaClient, ma_window: int = 200) -> tuple[str, float]:
    bars = client.get_bars(["SPY"], "Day", max(ma_window + 100, 400))
    if "SPY" not in bars or bars["SPY"].empty:
        logger.warning("No SPY data available")
        return "neutral", 0.0

    df = bars["SPY"].copy()
    close = df["close"]
    ma = close.rolling(ma_window).mean()
    latest_close = close.iloc[-1]
    latest_ma = ma.iloc[-1]

    if latest_close > latest_ma:
        return "long", 1.0
    elif latest_close < latest_ma:
        return "cash", 0.0
    else:
        return "neutral", 0.0


def execute(client: AlpacaClient, signal: str, current_position: float):
    trading_client = client.trading_client

    if signal == "long" and current_position == 0:
        account = trading_client.get_account()
        equity = float(account.equity)
        bars = client.get_bars(["SPY"], "Day", 5)
        if bars["SPY"].empty:
            return
        price = float(bars["SPY"]["close"].iloc[-1])
        qty = max(1, int(equity * 0.95 / price))
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        try:
            trading_client.submit_order(MarketOrderRequest(
                symbol="SPY", qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            ))
            logger.info("BUY %d SPY @ %.2f (above MA, entering market)", qty, price)
        except Exception as e:
            logger.warning("Failed to buy SPY: %s", e)

    elif signal == "cash" and current_position > 0:
        bars = client.get_bars(["SPY"], "Day", 5)
        if bars["SPY"].empty:
            return
        price = float(bars["SPY"]["close"].iloc[-1])
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        try:
            trading_client.submit_order(MarketOrderRequest(
                symbol="SPY", qty=int(current_position), side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            ))
            logger.info("SELL %d SPY @ %.2f (below MA, moving to cash)", int(current_position), price)
        except Exception as e:
            logger.warning("Failed to sell SPY: %s", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ma-window", type=int, default=200)
    args = parser.parse_args()

    client = AlpacaClient(paper=True)

    position = 0.0
    try:
        positions = client.trading_client.get_all_positions()
        for p in positions:
            if p.symbol == "SPY":
                position = abs(float(p.qty))
    except Exception:
        pass

    signal, _ = get_signal(client, args.ma_window)
    logger.info("SPY MA(%d) signal: %s | current SPY position: %.0f", args.ma_window, signal, position)

    execute(client, signal, position)


if __name__ == "__main__":
    main()
