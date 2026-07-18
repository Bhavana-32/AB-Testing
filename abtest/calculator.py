"""
Sample size and power calculations for two-proportion A/B tests.

Two effect-size conventions are implemented, because they disagree and the
disagreement matters:

1. "pooled"  -- the normal approximation on the raw proportion scale, using a
   pooled variance under H0 for the null term and separate variances for the
   alternative term. This is what most industry A/B calculators use
   (Evan Miller's, and the standard formula in Kohavi et al.).

2. "arcsine" -- Cohen's h, the variance-stabilising arcsine transform. This is
   what statsmodels' NormalIndPower + proportion_effectsize computes, and what
   R's pwr::pwr.2p.test computes.

They agree closely for moderate conversion rates and diverge at extreme ones.
`pooled` is the default because it matches the tools analysts actually compare
against; `arcsine` is exposed so the two can be shown side by side.

All functions here are pure -- no Streamlit imports -- so they can be tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize, stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

Method = Literal["pooled", "arcsine"]
Alternative = Literal["two-sided", "larger"]


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


def treatment_rate(baseline: float, mde: float, mde_type: str = "relative") -> float:
    """Convert a minimum detectable effect into an absolute treatment rate.

    `mde_type="relative"` means a proportional lift: baseline 0.05 with
    mde 0.10 -> 0.055. `mde_type="absolute"` means a percentage-point
    change: baseline 0.05 with mde 0.10 -> 0.15.

    This distinction is the single most common source of wrong answers in
    A/B sample size tools, so it is always explicit here -- there is no
    default that silently guesses.
    """
    _check_rate(baseline, "baseline")
    if mde_type == "relative":
        p2 = baseline * (1.0 + mde)
    elif mde_type == "absolute":
        p2 = baseline + mde
    else:
        raise ValueError(f"mde_type must be 'relative' or 'absolute', got {mde_type!r}")
    if not 0.0 < p2 < 1.0:
        raise ValueError(
            f"MDE implies a treatment rate of {p2:.4f}, which is outside (0, 1). "
            "Reduce the MDE or check whether you meant relative vs absolute."
        )
    return p2


def _check_rate(p: float, name: str) -> None:
    if not 0.0 < p < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {p}")


def _z_alpha(alpha: float, alternative: Alternative) -> float:
    """Critical value for the null. Two-sided splits alpha across both tails."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if alternative == "two-sided":
        return stats.norm.ppf(1.0 - alpha / 2.0)
    if alternative == "larger":
        return stats.norm.ppf(1.0 - alpha)
    raise ValueError(f"alternative must be 'two-sided' or 'larger', got {alternative!r}")


def normal_approximation_warnings(
    baseline: float, treatment: float, n_per_group: int, min_events: int = 10
) -> list[str]:
    """Flag inputs where the normal approximation underlying all of this breaks.

    Every formula in this module approximates a binomial with a normal. That
    approximation needs roughly 10 expected successes *and* 10 expected
    failures per group. Below that the calculator will still happily return a
    number, and the number will be wrong -- so the UI needs to say so rather
    than presenting a confident answer.
    """
    warnings: list[str] = []
    for label, p in (("control", baseline), ("treatment", treatment)):
        successes = n_per_group * p
        failures = n_per_group * (1.0 - p)
        if successes < min_events:
            warnings.append(
                f"Expected conversions in the {label} group is {successes:.1f} "
                f"(< {min_events}). The normal approximation is unreliable here; "
                "prefer an exact test."
            )
        if failures < min_events:
            warnings.append(
                f"Expected non-conversions in the {label} group is {failures:.1f} "
                f"(< {min_events}). The normal approximation is unreliable here."
            )
    return warnings


def bonferroni_alpha(alpha: float, n_variants: int) -> float:
    """Adjust alpha for comparing `n_variants - 1` treatments against one control.

    A 3-arm test (control + 2 treatments) makes 2 comparisons, so each is
    tested at alpha/2. Bonferroni is conservative -- Dunnett's correction is
    tighter for the many-vs-one-control case -- but it is transparent and
    never anti-conservative, which is the right trade for a planning tool.
    """
    if n_variants < 2:
        raise ValueError("n_variants must be at least 2 (control + 1 treatment)")
    n_comparisons = n_variants - 1
    return alpha / n_comparisons


# --------------------------------------------------------------------------
# Sample size and power
# --------------------------------------------------------------------------


def sample_size(
    baseline: float,
    treatment: float,
    alpha: float = 0.05,
    power: float = 0.80,
    method: Method = "pooled",
    alternative: Alternative = "two-sided",
) -> int:
    """Required sample size **per group**, rounded up.

    Assumes equal allocation between the two groups.
    """
    _check_rate(baseline, "baseline")
    _check_rate(treatment, "treatment")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    if baseline == treatment:
        raise ValueError("baseline and treatment rates are identical; MDE is zero")

    if method == "arcsine":
        h = proportion_effectsize(treatment, baseline)
        n = NormalIndPower().solve_power(
            effect_size=abs(h),
            nobs1=None,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative=alternative,
        )
        return int(math.ceil(n))

    if method == "pooled":
        z_a = _z_alpha(alpha, alternative)
        z_b = stats.norm.ppf(power)
        p_bar = (baseline + treatment) / 2.0
        null_sd = math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        alt_sd = math.sqrt(
            baseline * (1.0 - baseline) + treatment * (1.0 - treatment)
        )
        delta = abs(treatment - baseline)
        n = ((z_a * null_sd + z_b * alt_sd) / delta) ** 2
        return int(math.ceil(n))

    raise ValueError(f"method must be 'pooled' or 'arcsine', got {method!r}")


def achieved_power(
    baseline: float,
    treatment: float,
    n_per_group: int,
    alpha: float = 0.05,
    method: Method = "pooled",
    alternative: Alternative = "two-sided",
) -> float:
    """Power actually achieved at a given per-group sample size.

    Useful because `sample_size` rounds up, so realised power is always
    slightly above the target -- and because the reverse-mode question
    ("I have N users, what can I detect?") needs this as its objective.
    """
    _check_rate(baseline, "baseline")
    _check_rate(treatment, "treatment")
    if n_per_group < 1:
        raise ValueError("n_per_group must be at least 1")

    if method == "arcsine":
        h = proportion_effectsize(treatment, baseline)
        return float(
            NormalIndPower().solve_power(
                effect_size=abs(h),
                nobs1=n_per_group,
                alpha=alpha,
                power=None,
                ratio=1.0,
                alternative=alternative,
            )
        )

    if method == "pooled":
        z_a = _z_alpha(alpha, alternative)
        p_bar = (baseline + treatment) / 2.0
        null_sd = math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        alt_sd = math.sqrt(
            baseline * (1.0 - baseline) + treatment * (1.0 - treatment)
        )
        delta = abs(treatment - baseline)
        z = (delta * math.sqrt(n_per_group) - z_a * null_sd) / alt_sd
        return float(stats.norm.cdf(z))

    raise ValueError(f"method must be 'pooled' or 'arcsine', got {method!r}")


def detectable_effect(
    baseline: float,
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
    method: Method = "pooled",
    alternative: Alternative = "two-sided",
    direction: str = "increase",
) -> dict:
    """Reverse mode: given the sample you can actually get, what can you detect?

    This is the question analysts ask in practice ("we get 40k users a week
    and have three weeks") and the one most calculators refuse to answer.
    Solved numerically rather than by inverting the formula, because the
    pooled variance depends on the treatment rate we are solving for.

    Returns absolute and relative MDE alongside the implied treatment rate.
    """
    _check_rate(baseline, "baseline")
    if n_per_group < 1:
        raise ValueError("n_per_group must be at least 1")

    sign = 1.0 if direction == "increase" else -1.0
    upper = (1.0 - baseline) if direction == "increase" else baseline

    def objective(delta: float) -> float:
        p2 = baseline + sign * delta
        return achieved_power(baseline, p2, n_per_group, alpha, method, alternative) - power

    # Bracket the root. At delta -> 0 power collapses to alpha (below target);
    # at the largest admissible delta it should exceed the target.
    lo, hi = 1e-9, upper * (1.0 - 1e-9)
    if objective(hi) < 0:
        raise ValueError(
            f"Even the largest possible effect cannot reach {power:.0%} power "
            f"at n={n_per_group:,} per group. You need more traffic."
        )
    delta = optimize.brentq(objective, lo, hi, xtol=1e-10)
    p2 = baseline + sign * delta
    return {
        "absolute_mde": delta,
        "relative_mde": delta / baseline,
        "treatment_rate": p2,
    }


# --------------------------------------------------------------------------
# Duration
# --------------------------------------------------------------------------


@dataclass
class Duration:
    """Result of converting a required sample size into calendar time."""

    total_sample_needed: int
    usable_daily_traffic: float
    raw_days: float
    days: int
    weeks: float
    note: str


def duration(
    n_per_group: int,
    daily_traffic: float,
    n_groups: int = 2,
    allocation: float = 1.0,
    round_to_whole_weeks: bool = True,
) -> Duration:
    """Convert a per-group sample size into a test duration.

    `allocation` is the fraction of daily traffic actually entering the
    experiment -- rarely 1.0 in practice, and ignoring it is why estimates
    come in optimistic.

    Rounding up to whole weeks is the default deliberately. Conversion rates
    have strong day-of-week seasonality; a test that runs 10 days weights
    Mondays and Tuesdays double and can move the estimate more than the
    statistical uncertainty does.
    """
    if daily_traffic <= 0:
        raise ValueError("daily_traffic must be positive")
    if not 0.0 < allocation <= 1.0:
        raise ValueError("allocation must be in (0, 1]")

    total_needed = n_per_group * n_groups
    usable = daily_traffic * allocation
    raw_days = total_needed / usable

    if round_to_whole_weeks:
        days = int(math.ceil(raw_days / 7.0) * 7)
        note = "Rounded up to whole weeks to balance day-of-week seasonality."
    else:
        days = int(math.ceil(raw_days))
        note = (
            "Not rounded to whole weeks -- day-of-week effects will be "
            "unevenly weighted."
        )

    return Duration(
        total_sample_needed=total_needed,
        usable_daily_traffic=usable,
        raw_days=raw_days,
        days=days,
        weeks=days / 7.0,
        note=note,
    )


# --------------------------------------------------------------------------
# Curve for plotting
# --------------------------------------------------------------------------


def sample_size_curve(
    baseline: float,
    mde_values: np.ndarray,
    mde_type: str = "relative",
    alpha: float = 0.05,
    power: float = 0.80,
    method: Method = "pooled",
    alternative: Alternative = "two-sided",
) -> np.ndarray:
    """Required per-group sample size across a range of MDEs.

    Sample size scales roughly as 1/MDE^2, so this should be plotted on a
    log y-axis -- on linear axes the small-effect region is compressed into
    the top of the chart and the whole point of the plot is lost.
    """
    out = []
    for mde in mde_values:
        p2 = treatment_rate(baseline, float(mde), mde_type)
        out.append(sample_size(baseline, p2, alpha, power, method, alternative))
    return np.array(out, dtype=float)
