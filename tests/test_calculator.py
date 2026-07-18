"""
Validation tests for the sample size calculator.

The point of this file is not coverage for its own sake. It is to establish
that the numbers this tool produces agree with (a) the closed-form textbook
formulas, (b) an independent library, and (c) a Monte Carlo simulation that
makes no distributional shortcuts. Any one of those alone is easy to get
subtly wrong.
"""

import math

import numpy as np
import pytest
from scipy import stats

from abtest.calculator import (
    achieved_power,
    bonferroni_alpha,
    detectable_effect,
    duration,
    normal_approximation_warnings,
    sample_size,
    sample_size_curve,
    treatment_rate,
)


# --------------------------------------------------------------------------
# Input handling -- relative vs absolute MDE
# --------------------------------------------------------------------------


def test_relative_mde():
    assert treatment_rate(0.05, 0.10, "relative") == pytest.approx(0.055)


def test_absolute_mde():
    assert treatment_rate(0.05, 0.10, "absolute") == pytest.approx(0.15)


def test_relative_and_absolute_differ_hugely():
    """The distinction is worth ~25x in sample size, hence the explicit flag."""
    n_rel = sample_size(0.05, treatment_rate(0.05, 0.10, "relative"))
    n_abs = sample_size(0.05, treatment_rate(0.05, 0.10, "absolute"))
    assert n_rel / n_abs > 20


def test_impossible_mde_raises():
    with pytest.raises(ValueError, match="outside"):
        treatment_rate(0.9, 0.5, "absolute")


def test_bad_rates_raise():
    with pytest.raises(ValueError):
        sample_size(0.0, 0.05)
    with pytest.raises(ValueError):
        sample_size(0.05, 1.0)
    with pytest.raises(ValueError, match="identical"):
        sample_size(0.05, 0.05)


# --------------------------------------------------------------------------
# Agreement with closed-form theory
# --------------------------------------------------------------------------


def test_arcsine_matches_cohen_closed_form():
    """statsmodels' NormalIndPower should reproduce n = 2(z_a + z_b)^2 / h^2.

    The factor of 2 is easy to drop and worth spelling out. The arcsine
    transform phi = 2*arcsin(sqrt(p)) has Var(phi) ~ 1/n, so the difference
    between two independent arms has variance 2/n, and the test statistic is
    h / sqrt(2/n). Solving for n therefore carries the 2.

    This is the same closed form R's pwr::pwr.2p.test uses, so agreement here
    means our arcsine numbers match the standard R implementation.
    """
    p1, p2 = 0.50, 0.55
    h = 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))
    z_a = stats.norm.ppf(1 - 0.05 / 2)
    z_b = stats.norm.ppf(0.80)
    expected = 2 * (z_a + z_b) ** 2 / h**2

    n = sample_size(p1, p2, alpha=0.05, power=0.80, method="arcsine")
    # Not exact equality: solve_power's two-sided power function retains the
    # far-tail term that the closed form drops. The gap is <0.1%.
    assert n == pytest.approx(expected, rel=0.002)


def test_pooled_matches_hand_computation():
    """Independent transcription of the pooled normal-approximation formula."""
    p1, p2 = 0.10, 0.12
    z_a = stats.norm.ppf(1 - 0.05 / 2)
    z_b = stats.norm.ppf(0.80)
    p_bar = (p1 + p2) / 2
    numerator = (
        z_a * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    expected = numerator / (p2 - p1) ** 2

    assert sample_size(p1, p2) == math.ceil(expected)


@pytest.mark.parametrize(
    "p1,p2",
    [
        (0.40, 0.45),  # moderate rate, small effect
        (0.10, 0.12),  # typical e-commerce conversion rate
        (0.05, 0.055),  # low rate, 10% relative lift
        (0.005, 0.006),  # very low rate, 20% relative lift
    ],
)
def test_pooled_and_arcsine_agree_for_realistic_effects(p1, p2):
    """For the small relative effects real A/B tests target, the two
    conventions agree to within a fraction of a percent -- including at very
    low conversion rates. Divergence is driven by the *size of the relative
    effect*, not by how extreme the base rate is.
    """
    n_pooled = sample_size(p1, p2, method="pooled")
    n_arcsine = sample_size(p1, p2, method="arcsine")
    assert abs(n_pooled - n_arcsine) / n_pooled < 0.005


def test_pooled_and_arcsine_diverge_for_large_relative_effects():
    """A 5x lift is where the arcsine transform's curvature actually bites.

    Kept as a regression guard: if this ever stops failing to differ, the two
    methods have silently become the same thing and the README is then wrong.
    """
    n_pooled = sample_size(0.01, 0.05, method="pooled")
    n_arcsine = sample_size(0.01, 0.05, method="arcsine")
    assert abs(n_pooled - n_arcsine) / n_pooled > 0.05


# --------------------------------------------------------------------------
# Internal consistency
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pooled", "arcsine"])
@pytest.mark.parametrize("power", [0.70, 0.80, 0.90])
def test_sample_size_and_power_are_inverses(method, power):
    n = sample_size(0.10, 0.11, alpha=0.05, power=power, method=method)
    realised = achieved_power(0.10, 0.11, n, alpha=0.05, method=method)
    # Rounding up means realised power is at least the target, never far above.
    assert power <= realised < power + 0.005


@pytest.mark.parametrize("method", ["pooled", "arcsine"])
def test_detectable_effect_round_trips(method):
    n = 20_000
    res = detectable_effect(0.10, n, alpha=0.05, power=0.80, method=method)
    realised = achieved_power(0.10, res["treatment_rate"], n, method=method)
    assert realised == pytest.approx(0.80, abs=1e-6)


def test_detectable_effect_raises_when_truly_underpowered():
    with pytest.raises(ValueError, match="more traffic"):
        detectable_effect(0.10, 3)


def test_detectable_effect_at_tiny_n_is_flagged_as_unreliable():
    """At n=5 the solver still returns an answer -- an 88% treatment rate.

    Arithmetically valid, statistically meaningless: the normal approximation
    has no business being applied to 5 users. The guardrail, not the solver,
    is what stops a user trusting this.
    """
    res = detectable_effect(0.10, 5)
    assert res["treatment_rate"] > 0.8
    assert normal_approximation_warnings(0.10, res["treatment_rate"], 5)


def test_no_warnings_for_well_powered_realistic_inputs():
    n = sample_size(0.10, 0.12)
    assert normal_approximation_warnings(0.10, 0.12, n) == []


def test_warns_on_rare_events():
    warnings = normal_approximation_warnings(0.001, 0.0012, 5_000)
    assert any("normal approximation" in w for w in warnings)


def test_smaller_effects_need_more_data():
    sizes = sample_size_curve(0.10, np.array([0.20, 0.10, 0.05, 0.02]))
    assert np.all(np.diff(sizes) > 0)


def test_sample_size_scales_as_inverse_square_of_mde():
    """Halving the MDE should roughly quadruple the sample size.

    This is the relationship the power curve chart exists to communicate,
    so it is worth asserting rather than assuming.
    """
    n_big = sample_size(0.10, treatment_rate(0.10, 0.10, "relative"))
    n_small = sample_size(0.10, treatment_rate(0.10, 0.05, "relative"))
    assert 3.8 < n_small / n_big < 4.2


def test_more_power_needs_more_data():
    assert sample_size(0.10, 0.11, power=0.90) > sample_size(0.10, 0.11, power=0.80)


def test_stricter_alpha_needs_more_data():
    assert sample_size(0.10, 0.11, alpha=0.01) > sample_size(0.10, 0.11, alpha=0.05)


def test_one_sided_needs_less_data_than_two_sided():
    n_one = sample_size(0.10, 0.11, alternative="larger")
    n_two = sample_size(0.10, 0.11, alternative="two-sided")
    assert n_one < n_two


# --------------------------------------------------------------------------
# Monte Carlo check -- does the computed sample size actually deliver 80%?
# --------------------------------------------------------------------------


def test_computed_sample_size_delivers_requested_power_by_simulation():
    """The strongest available check: no formula reused, just simulated data.

    Draws 20,000 experiments at the recommended sample size and counts how
    often the test correctly detects the real effect. Should land on 80%.
    """
    p1, p2 = 0.10, 0.12
    n = sample_size(p1, p2, alpha=0.05, power=0.80)

    rng = np.random.default_rng(0)
    sims = 20_000
    c1 = rng.binomial(n, p1, sims)
    c2 = rng.binomial(n, p2, sims)

    p_hat1, p_hat2 = c1 / n, c2 / n
    p_pool = (c1 + c2) / (2 * n)
    se = np.sqrt(p_pool * (1 - p_pool) * (2 / n))
    z = (p_hat2 - p_hat1) / se
    empirical_power = float((np.abs(z) > stats.norm.ppf(0.975)).mean())

    assert empirical_power == pytest.approx(0.80, abs=0.02)


# --------------------------------------------------------------------------
# Multiple comparisons and duration
# --------------------------------------------------------------------------


def test_bonferroni_splits_alpha_across_comparisons():
    assert bonferroni_alpha(0.05, 2) == pytest.approx(0.05)  # 1 comparison
    assert bonferroni_alpha(0.05, 3) == pytest.approx(0.025)  # 2 comparisons
    assert bonferroni_alpha(0.05, 5) == pytest.approx(0.0125)  # 4 comparisons


def test_more_variants_need_more_data_per_arm():
    n2 = sample_size(0.10, 0.11, alpha=bonferroni_alpha(0.05, 2))
    n4 = sample_size(0.10, 0.11, alpha=bonferroni_alpha(0.05, 4))
    assert n4 > n2


def test_duration_rounds_up_to_whole_weeks():
    d = duration(n_per_group=10_000, daily_traffic=5_000, n_groups=2)
    assert d.raw_days == pytest.approx(4.0)
    assert d.days == 7
    assert d.weeks == 1.0


def test_duration_respects_allocation():
    full = duration(10_000, 5_000, allocation=1.0, round_to_whole_weeks=False)
    half = duration(10_000, 5_000, allocation=0.5, round_to_whole_weeks=False)
    assert half.days == 2 * full.days


def test_duration_accounts_for_extra_arms():
    two = duration(10_000, 5_000, n_groups=2, round_to_whole_weeks=False)
    three = duration(10_000, 5_000, n_groups=3, round_to_whole_weeks=False)
    assert three.raw_days == pytest.approx(1.5 * two.raw_days)


def test_duration_rejects_bad_inputs():
    with pytest.raises(ValueError):
        duration(1000, 0)
    with pytest.raises(ValueError):
        duration(1000, 500, allocation=0)
