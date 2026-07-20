import logging
from datetime import datetime

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logger = logging.getLogger(__name__)


class TradingExecutor:
    def __init__(self, client: TradingClient, config: dict):
        self.client = client
        self.config = config
        self.max_position_pct = config.get("max_position_pct", 0.1)
        self.stop_loss_pct = config.get("stop_loss_pct", 0.02)
        self.take_profit_pct = config.get("take_profit_pct", 0.05)
        self.max_open_positions = config.get("max_open_positions", 5)

    def get_cash(self) -> float:
        account = self.client.get_account()
        return float(account.cash)

    def get_positions(self) -> dict[str, float]:
        positions = self.client.get_all_positions()
        return {p.symbol: float(p.qty) for p in positions}

    def execute_signals(
        self, predictions: dict[str, float], prices: dict[str, float]
    ) -> list[str]:
        orders_placed = []
        cash = self.get_cash()
        current_positions = self.get_positions()

        for symbol, signal in predictions.items():
            current_qty = current_positions.get(symbol, 0)

            if signal > 0 and current_qty <= 0:
                if len(current_positions) >= self.max_open_positions:
                    logger.info("Max open positions reached, skipping %s", symbol)
                    continue

                price = prices.get(symbol)
                if price is None or price <= 0:
                    continue

                allocation = cash * self.max_position_pct
                qty = max(1, int(allocation / price))

                order = self.client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                orders_placed.append(f"Bought {qty} {symbol} @ {price:.2f}")
                logger.info("Bought %d %s @ %.2f", qty, symbol, price)

            elif signal < 0 and current_qty > 0:
                order = self.client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=current_qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                orders_placed.append(f"Sold {current_qty} {symbol}")
                logger.info("Sold %f %s", current_qty, symbol)

        return orders_placed

    def close_all_positions(self) -> list[str]:
        positions = self.client.get_all_positions()
        closed = []
        for pos in positions:
            order = self.client.submit_order(
                MarketOrderRequest(
                    symbol=pos.symbol,
                    qty=float(pos.qty),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            closed.append(f"Closed {pos.symbol}")
        return closed
