"""Risk management with configurable limits.

This module implements risk management checks that are applied
before every trade to ensure compliance with portfolio-level limits.
"""

import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk manager for pair trading strategies.

    Checks all risk limits before approving a trade signal.
    All limits are configurable via config/default.yaml.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
    ):
        """Initialize the risk manager.

        Args:
            config: Configuration dict with risk limits. If None, loads from default.yaml.
        """
        self.config = config or self._load_default_config()

        # Risk limits
        self.max_position_pct = self.config.get("max_position_pct", 0.20)
        self.max_gross_exposure = self.config.get("max_gross_exposure", 2.0)
        self.max_net_exposure = self.config.get("max_net_exposure", 0.50)
        self.max_pair_exposure = self.config.get("max_pair_exposure", 0.40)
        self.max_portfolio_exposure = self.config.get("max_portfolio_exposure", 1.5)
        self.stop_loss_z = self.config.get("stop_loss_z", 3.0)
        self.max_holding_period_bars = self.config.get("max_holding_period_bars", 100)

        logger.info(
            f"Initialized RiskManager: max_position={self.max_position_pct}, "
            f"max_gross={self.max_gross_exposure}, max_net={self.max_net_exposure}"
        )

    def _load_default_config(self) -> dict[str, Any]:
        """Load default configuration from config/default.yaml."""
        try:
            import yaml
            from pathlib import Path

            config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    return config.get("risk_limits", {})
        except Exception as e:
            logger.warning(f"Could not load default config: {e}")

        return {
            "max_position_pct": 0.20,
            "max_gross_exposure": 2.0,
            "max_net_exposure": 0.50,
            "max_pair_exposure": 0.40,
            "max_portfolio_exposure": 1.5,
            "stop_loss_z": 3.0,
            "max_holding_period_bars": 100,
        }

    def check_signal(
        self,
        signal: int,
        current_positions: dict[str, dict[str, float]],
        portfolio_equity: float,
        market_data: dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check if a signal passes all risk limits.

        Args:
            signal: Trade signal (1 = long A/short B, -1 = short A/long B, 0 = no signal).
            current_positions: Dict mapping pair name to position dict with
                              'shares_a', 'shares_b', 'dollar_value'.
            portfolio_equity: Current total portfolio equity.
            market_data: Dict with current market data:
                - zscore: Current z-score of the spread
                - prices: Dict of current prices
                - volatilities: Dict of current volatilities
                - pair_name: Name of the pair

        Returns:
            Tuple of (approved: bool, reason: str).
            If approved is False, reason contains the rejection reason.
        """
        if signal == 0:
            return True, "no_signal"

        # Extract market data
        zscore = market_data.get("zscore", 0.0)
        pair_name = market_data.get("pair_name", "UNKNOWN")
        prices = market_data.get("prices", {})
        price_a = prices.get("a", 0.0)
        price_b = prices.get("b", 0.0)

        # Calculate current position values
        current_position = current_positions.get(pair_name, {})
        current_shares_a = current_position.get("shares_a", 0.0)
        current_shares_b = current_position.get("shares_b", 0.0)
        current_dollar_value = current_position.get("dollar_value", 0.0)

        # Calculate proposed trade value
        if price_a > 0 and price_b > 0:
            # Estimate trade value (notional for one leg)
            trade_value = portfolio_equity * self.max_position_pct
            proposed_shares_a = trade_value / price_a
            proposed_shares_b = trade_value / price_b
            proposed_dollar_value = trade_value
        else:
            return False, "invalid_prices"

        # Check 1: Maximum position size (per leg)
        max_position_value = portfolio_equity * self.max_position_pct
        if proposed_dollar_value > max_position_value:
            return False, f"max_position_size: proposed={proposed_dollar_value:.2f}, max={max_position_value:.2f}"

        # Check 2: Maximum gross exposure
        # Gross exposure = sum of absolute values of all positions
        total_gross = sum(
            abs(pos.get("dollar_value", 0)) for pos in current_positions.values()
        )
        proposed_gross = total_gross + proposed_dollar_value
        max_gross_exposure_value = portfolio_equity * self.max_gross_exposure
        if proposed_gross > max_gross_exposure_value:
            return False, f"max_gross_exposure: proposed={proposed_gross:.2f}, max={max_gross_exposure_value:.2f}"

        # Check 3: Maximum net exposure
        # Net exposure = sum of signed values of all positions
        total_net = sum(
            pos.get("dollar_value", 0) for pos in current_positions.values()
        )
        # For a pair trade, one leg is long and one is short
        # The net exposure is the difference
        sign = 1 if signal == 1 else -1
        proposed_net = total_net + sign * proposed_dollar_value * 0.5  # Approximate
        max_net_exposure_value = portfolio_equity * self.max_net_exposure
        if abs(proposed_net) > max_net_exposure_value:
            return False, f"max_net_exposure: proposed={abs(proposed_net):.2f}, max={max_net_exposure_value:.2f}"

        # Check 4: Maximum pair exposure
        new_pair_exposure = current_dollar_value + proposed_dollar_value
        max_pair_exposure_value = portfolio_equity * self.max_pair_exposure
        if new_pair_exposure > max_pair_exposure_value:
            return False, f"max_pair_exposure: proposed={new_pair_exposure:.2f}, max={max_pair_exposure_value:.2f}"

        # Check 5: Maximum portfolio exposure (overall limit)
        total_portfolio_exposure = sum(
            abs(pos.get("dollar_value", 0)) for pos in current_positions.values()
        ) + proposed_dollar_value
        max_portfolio_exposure_value = portfolio_equity * self.max_portfolio_exposure
        if total_portfolio_exposure > max_portfolio_exposure_value:
            return False, f"max_portfolio_exposure: proposed={total_portfolio_exposure:.2f}, max={max_portfolio_exposure_value:.2f}"

        # Check 6: Stop loss z-score
        if abs(zscore) > self.stop_loss_z:
            return False, f"stop_loss_z: zscore={abs(zscore):.2f} exceeds {self.stop_loss_z}"

        # All checks passed
        return True, "approved"

    def apply_volatility_scaling(
        self,
        signal_size: float,
        current_vol: float,
        target_vol: float,
    ) -> float:
        """Apply volatility scaling to position size.

        Scales the position size to achieve target volatility.

        Args:
            signal_size: Base signal size (notional value).
            current_vol: Current portfolio/strategy volatility.
            target_vol: Target volatility.

        Returns:
            Adjusted position size.
        """
        if current_vol <= 0:
            logger.warning("Invalid current volatility, returning base signal size")
            return signal_size

        # Volatility scaling formula
        # If current volatility is higher than target, reduce position
        scale_factor = target_vol / current_vol

        # Bound the scale factor to prevent extreme adjustments
        scale_factor = max(0.1, min(scale_factor, 3.0))

        adjusted_size = signal_size * scale_factor

        logger.debug(
            f"Volatility scaling: current_vol={current_vol:.4f}, "
            f"target_vol={target_vol:.4f}, scale={scale_factor:.4f}, "
            f"adjusted_size={adjusted_size:.2f}"
        )

        return adjusted_size

    def drawdown_reduction(
        self,
        equity_curve: pd.Series,
        drawdown_threshold: float = 0.10,
    ) -> float:
        """Calculate position reduction factor based on drawdown.

        Reduces position sizes when drawdown exceeds threshold.

        Args:
            equity_curve: Series of equity values.
            drawdown_threshold: Drawdown threshold (e.g., 0.10 for 10%).

        Returns:
            Position reduction factor (0.0 to 1.0).
            1.0 means no reduction, 0.5 means reduce positions by 50%.
        """
        if len(equity_curve) < 2:
            return 1.0

        # Calculate current drawdown
        running_max = equity_curve.expanding().max()
        current_drawdown = (equity_curve.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]

        if current_drawdown >= 0:
            # Not in drawdown, full position
            return 1.0

        drawdown_pct = abs(current_drawdown)

        if drawdown_pct < drawdown_threshold:
            # Below threshold, no reduction
            return 1.0

        # Linear reduction from threshold to max drawdown
        # At threshold, reduction factor = 1.0
        # At some max drawdown (e.g., 50%), reduction factor = 0.0
        max_drawdown = 0.50

        if drawdown_pct >= max_drawdown:
            return 0.0

        # Linear interpolation
        reduction = (drawdown_pct - drawdown_threshold) / (max_drawdown - drawdown_threshold)
        reduction_factor = 1.0 - reduction

        # Bound the result
        reduction_factor = max(0.0, min(1.0, reduction_factor))

        logger.info(
            f"Drawdown reduction: drawdown={drawdown_pct:.2%}, "
            f"threshold={drawdown_threshold:.2%}, factor={reduction_factor:.4f}"
        )

        return reduction_factor

    def check_holding_period(
        self,
        entry_time: pd.Timestamp,
        current_time: pd.Timestamp,
        timeframe_bars_per_day: int = 24,
    ) -> Tuple[bool, str]:
        """Check if position exceeds maximum holding period.

        Args:
            entry_time: Position entry timestamp.
            current_time: Current timestamp.
            timeframe_bars_per_day: Number of bars per day (for 1h = 24).

        Returns:
            Tuple of (within_limits: bool, reason: str).
        """
        if entry_time is None:
            return True, "no_position"

        holding_period = current_time - entry_time
        holding_days = holding_period.total_seconds() / (24 * 3600)

        # Convert max holding period to days
        max_holding_days = self.max_holding_period_bars / timeframe_bars_per_day

        if holding_days > max_holding_days:
            return False, f"max_holding_period: held={holding_days:.1f}d, max={max_holding_days:.1f}d"

        return True, "within_limits"

    def get_risk_summary(
        self,
        positions: dict[str, dict[str, float]],
        portfolio_equity: float,
    ) -> dict[str, Any]:
        """Get current risk summary.

        Args:
            positions: Dict of current positions.
            portfolio_equity: Current portfolio equity.

        Returns:
            Dict with risk metrics.
        """
        total_gross = sum(abs(pos.get("dollar_value", 0)) for pos in positions.values())
        total_net = sum(pos.get("dollar_value", 0) for pos in positions.values())

        return {
            "portfolio_equity": portfolio_equity,
            "total_gross_exposure": total_gross,
            "total_net_exposure": total_net,
            "gross_exposure_pct": total_gross / portfolio_equity if portfolio_equity > 0 else 0,
            "net_exposure_pct": total_net / portfolio_equity if portfolio_equity > 0 else 0,
            "n_active_positions": len(positions),
            "max_position_pct": self.max_position_pct,
            "max_gross_exposure": self.max_gross_exposure,
            "max_net_exposure": self.max_net_exposure,
        }
