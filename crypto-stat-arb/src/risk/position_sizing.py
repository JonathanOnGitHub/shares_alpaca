"""Position sizing for all 4 hedging variants.

This module implements position sizing for pair trading strategies:
1. Equal-dollar hedge: Equal dollar value in each leg
2. OLS hedge ratio: Uses OLS-estimated beta for hedging
3. Volatility-adjusted hedge: Adjusts for different volatilities
4. Beta-neutral hedge: Accounts for existing portfolio beta
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def equal_dollar_hedge(
    notional: float,
    price_a: float,
    price_b: float,
) -> Tuple[float, float]:
    """Calculate shares for equal-dollar hedging.

    Allocates equal dollar value to each leg of the pair trade.

    Args:
        notional: Total notional value to allocate.
        price_a: Price of asset A.
        price_b: Price of asset B.

    Returns:
        Tuple of (shares_a, shares_b).
    """
    if price_a <= 0 or price_b <= 0:
        logger.warning("Invalid prices for equal dollar hedge")
        return 0.0, 0.0

    # Equal dollar allocation per leg
    dollar_per_leg = notional / 2

    shares_a = dollar_per_leg / price_a
    shares_b = dollar_per_leg / price_b

    logger.debug(
        f"Equal dollar hedge: notional={notional:.2f}, "
        f"shares_a={shares_a:.6f}, shares_b={shares_b:.6f}"
    )

    return shares_a, shares_b


def ols_hedge_ratio(
    notional: float,
    price_a: float,
    price_b: float,
    beta: float,
) -> Tuple[float, float]:
    """Calculate shares using OLS hedge ratio.

    Uses the OLS-estimated beta to determine the hedge ratio.
    The spread is: log(P_A) - beta * log(P_B)

    Args:
        notional: Total notional value to allocate.
        price_a: Price of asset A.
        price_b: Price of asset B.
        beta: OLS-estimated hedge ratio (beta coefficient).

    Returns:
        Tuple of (shares_a, shares_b).
    """
    if price_a <= 0 or price_b <= 0:
        logger.warning("Invalid prices for OLS hedge ratio")
        return 0.0, 0.0

    if beta <= 0:
        logger.warning(f"Invalid beta {beta} for OLS hedge ratio, using 1.0")
        beta = 1.0

    # Dollar-neutral position sizing
    # shares_a * price_a = shares_b * price_b * beta
    # shares_a * price_a + shares_b * price_b = 2 * notional / 2 = notional

    # Total notional split between the two legs
    # Let x = dollar value in A, then dollar value in B = notional - x
    # shares_a = x / price_a
    # shares_b = (notional - x) / price_b
    # Hedge condition: x = (notional - x) * beta
    # x = beta * notional - beta * x
    # x * (1 + beta) = beta * notional
    # x = beta * notional / (1 + beta)

    dollar_a = beta * notional / (1 + beta)
    dollar_b = notional - dollar_a

    shares_a = dollar_a / price_a
    shares_b = dollar_b / price_b

    logger.debug(
        f"OLS hedge ratio: notional={notional:.2f}, beta={beta:.4f}, "
        f"shares_a={shares_a:.6f}, shares_b={shares_b:.6f}"
    )

    return shares_a, shares_b


def volatility_adjusted_hedge(
    notional: float,
    price_a: float,
    price_b: float,
    vol_a: float,
    vol_b: float,
    target_vol: float = 1.0,
) -> Tuple[float, float]:
    """Calculate shares using volatility-adjusted hedging.

    Adjusts position sizes to equalize the volatility contribution
    of each leg to the portfolio.

    Args:
        notional: Total notional value to allocate.
        price_a: Price of asset A.
        price_b: Price of asset B.
        vol_a: Volatility (standard deviation) of asset A.
        vol_b: Volatility (standard deviation) of asset B.
        target_vol: Target portfolio volatility.

    Returns:
        Tuple of (shares_a, shares_b).
    """
    if price_a <= 0 or price_b <= 0:
        logger.warning("Invalid prices for volatility-adjusted hedge")
        return 0.0, 0.0

    if vol_a <= 0 or vol_b <= 0:
        logger.warning("Invalid volatilities for volatility-adjusted hedge")
        return equal_dollar_hedge(notional, price_a, price_b)

    # Calculate dollar volatilities
    vol_dollar_a = vol_a * price_a  # Dollar volatility per share
    vol_dollar_b = vol_b * price_b

    # Target dollar volatility per leg
    target_vol_dollar = target_vol * notional / 2

    # Number of shares to achieve target volatility
    shares_a = target_vol_dollar / vol_dollar_a if vol_dollar_a > 0 else notional / (2 * price_a)
    shares_b = target_vol_dollar / vol_dollar_b if vol_dollar_b > 0 else notional / (2 * price_b)

    # Scale to notional constraint
    actual_notional = shares_a * price_a + shares_b * price_b
    if actual_notional > 0:
        scale = notional / actual_notional
        shares_a *= scale
        shares_b *= scale
    else:
        shares_a, shares_b = equal_dollar_hedge(notional, price_a, price_b)

    logger.debug(
        f"Volatility-adjusted hedge: notional={notional:.2f}, "
        f"vol_a={vol_a:.4f}, vol_b={vol_b:.4f}, target={target_vol:.4f}, "
        f"shares_a={shares_a:.6f}, shares_b={shares_b:.6f}"
    )

    return shares_a, shares_b


def beta_neutral_hedge(
    notional: float,
    price_a: float,
    price_b: float,
    beta: float,
    beta_existing: float = 0.0,
) -> Tuple[float, float]:
    """Calculate shares for beta-neutral hedging.

    Adjusts the hedge ratio to neutralize the portfolio beta
    when adding this pair trade.

    Args:
        notional: Total notional value to allocate.
        price_a: Price of asset A.
        price_b: Price of asset B.
        beta: Beta of asset A relative to asset B (from regression).
        beta_existing: Existing portfolio beta to neutralize.

    Returns:
        Tuple of (adjusted_shares_a, adjusted_shares_b).
    """
    if price_a <= 0 or price_b <= 0:
        logger.warning("Invalid prices for beta-neutral hedge")
        return 0.0, 0.0

    if beta <= 0:
        logger.warning(f"Invalid beta {beta} for beta-neutral hedge, using 1.0")
        beta = 1.0

    # First calculate base OLS hedge ratio
    base_shares_a, base_shares_b = ols_hedge_ratio(notional, price_a, price_b, beta)

    # Calculate position betas
    # Portfolio beta = sum(position_beta * position_value) / total_value
    # For beta neutrality: new_portfolio_beta = 0
    #
    # Let w_a = base_shares_a * price_a (dollar value in A)
    # Let w_b = base_shares_b * price_b (dollar value in B)
    # Total = w_a + w_b
    #
    # Pair beta contribution = w_a * 1 + w_b * beta (A has beta 1 to itself)
    # We want to scale positions so that the beta contribution is adjusted
    # to offset existing portfolio beta

    dollar_a = base_shares_a * price_a
    dollar_b = base_shares_b * price_b

    # The beta contribution of the pair trade
    pair_beta = (dollar_a * 1 + dollar_b * beta) / (dollar_a + dollar_b) if (dollar_a + dollar_b) > 0 else 1.0

    # Scale to achieve beta neutrality
    # If existing portfolio has beta, we need to adjust
    if abs(beta_existing) > 0.01:
        # Scale factor to reduce portfolio beta
        scale = 1.0 / (1.0 + beta_existing / pair_beta) if abs(pair_beta) > 0.01 else 1.0
        scale = max(0.1, min(scale, 2.0))  # Bound the scale factor
    else:
        scale = 1.0

    adjusted_shares_a = base_shares_a * scale
    adjusted_shares_b = base_shares_b * scale

    logger.debug(
        f"Beta-neutral hedge: notional={notional:.2f}, beta={beta:.4f}, "
        f"beta_existing={beta_existing:.4f}, scale={scale:.4f}, "
        f"shares_a={adjusted_shares_a:.6f}, shares_b={adjusted_shares_b:.6f}"
    )

    return adjusted_shares_a, adjusted_shares_b
