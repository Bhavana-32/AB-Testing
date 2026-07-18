"""
Validation tests for the peeking simulation.

The simulation is checked against two independent published sources:

  * Armitage, McPherson & Rowe (1969) for the inflated false positive rate
    under repeated significance testing.
  * Pocock (1977) for the corrected constant boundary that fixes it.

If the simulation reproduces both tables it is almost certainly right. If it
reproduces neither, no amount of plotting will make the app credible.
"""

import numpy as np
import pytest
from scipy import stats

from abtest.simulation import (
    ARMITAGE_1969,
    POCOCK_NOMINAL_ALPHA,
    calibrate_pocock,
    false_positive_rate,
    first_crossing,
    peeking_curve,
    power_under_boundary,
    recovery_rate,
    simulate_z_trajectories,
)

Z_95 = stats.norm.ppf(0.975)

# Large n and p near 0.5 keep the normal approximation tight, so any
# disagreement with the published tables is attributable to the simulation
# logic rather than to binomial discreteness at small counts.
N_PER_ARM = 10_000
P = 0.5
N_SIMS = 40_000


@pytest.fixture(scope="module")
def null_z():
    """A/A trajectories with 20 looks, reused across tests."""
    return simulate_z_trajectories(
        n_per_arm=N_PER_ARM, n_looks=20, p_control=P, n_sims=N_SIMS, seed=1
    )


# --------------------------------------------------------------------------
# Sanity: the simulation itself
# --------------------------------------------------------------------------


def test_trajectory_shape():
    z = simulate_z_trajectories(1000, 5, 0.2, n_sims=100)
    assert z.shape == (100, 5)


def test_no_nans_even_with_zero_conversions():
    """Very low rates produce looks with zero conversions in both arms."""
    z = simulate_z_trajectories(1000, 10, 0.0005, n_sims=500, seed=3)
    assert np.isfinite(z).all()


def test_null_z_is_standard_normal_at_final_look(null_z):
    final = null_z[:, -1]
    assert final.mean() == pytest.approx(0.0, abs=0.02)
    assert final.std() == pytest.approx(1.0, abs=0.02)


def test_true_effect_shifts_z_positive():
    z = simulate_z_trajectories(20_000, 5, 0.10, p_treatment=0.12, n_sims=2_000, seed=5)
    assert z[:, -1].mean() > 3.0


def test_looks_are_correlated_not_independent():
    """Data accumulates, so adjacent looks must be highly correlated.

    A common way to get this simulation wrong is to redraw fresh samples at
    each look, which makes the looks independent and produces a *different*
    (and wrong) false positive curve.
    """
    z = simulate_z_trajectories(10_000, 10, 0.3, n_sims=5_000, seed=11)
    r = np.corrcoef(z[:, -2], z[:, -1])[0, 1]
    assert r > 0.85


# --------------------------------------------------------------------------
# The peeking problem -- validated against Armitage et al. (1969)
# --------------------------------------------------------------------------


def test_single_look_gives_nominal_alpha(null_z):
    """One look at the end = a normal fixed-horizon test = 5%."""
    fpr = false_positive_rate(null_z[:, -1:], Z_95)
    assert fpr == pytest.approx(0.05, abs=0.005)


@pytest.mark.parametrize("k", [2, 3, 5, 10, 20])
def test_peeking_reproduces_armitage_1969(k):
    z = simulate_z_trajectories(
        n_per_arm=N_PER_ARM, n_looks=k, p_control=P, n_sims=N_SIMS, seed=100 + k
    )
    fpr = false_positive_rate(z, Z_95)
    assert fpr == pytest.approx(ARMITAGE_1969[k], abs=0.015)


def test_false_positive_rate_increases_with_peeking():
    results = peeking_curve(
        n_per_arm=N_PER_ARM, p_control=P, look_counts=(1, 2, 5, 10, 20), n_sims=20_000
    )
    rates = [r.false_positive_rate for r in results]
    assert rates == sorted(rates)
    assert rates[0] == pytest.approx(0.05, abs=0.01)
    assert rates[-1] > 0.20


def test_daily_peeking_over_a_month_is_badly_inflated():
    """The headline claim of the project, stated as an assertion."""
    z = simulate_z_trajectories(30_000, 30, P, n_sims=20_000, seed=42)
    assert false_positive_rate(z, Z_95) > 0.25


# --------------------------------------------------------------------------
# When and how the false positives happen
# --------------------------------------------------------------------------


def test_false_positives_cluster_early(null_z):
    """Crossings concentrate in the early looks, when estimates are noisiest.

    This is the mechanism behind the problem: early looks have wide
    confidence intervals, so a random excursion is more likely to reach the
    threshold exactly when the team is most tempted to stop.
    """
    idx = first_crossing(null_z, Z_95)
    crossings = idx[idx >= 0]
    first_half = (crossings < 10).mean()
    assert first_half > 0.5


def test_most_early_winners_do_not_survive(null_z):
    """Most tests that cross the threshold end up non-significant anyway."""
    assert recovery_rate(null_z, Z_95) > 0.5


# --------------------------------------------------------------------------
# The fix -- validated against Pocock (1977)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5, 10])
def test_calibrated_boundary_matches_pocock_table(k):
    b = calibrate_pocock(
        n_per_arm=N_PER_ARM, n_looks=k, p_control=P, alpha=0.05, n_sims=60_000, seed=7
    )
    assert b.nominal_alpha == pytest.approx(POCOCK_NOMINAL_ALPHA[k], abs=0.0025)


def test_pocock_boundary_is_stricter_than_naive():
    b = calibrate_pocock(N_PER_ARM, 10, P, n_sims=40_000)
    assert b.z_critical > Z_95
    assert b.nominal_alpha < 0.05


def test_boundary_controls_error_out_of_sample():
    """Calibrate on one set of simulations, verify on a fresh, unseen set.

    Checking the boundary against the same data used to fit it would be
    circular -- the quantile matches by construction. This is the test that
    actually establishes the correction works.
    """
    b = calibrate_pocock(N_PER_ARM, 10, P, alpha=0.05, n_sims=60_000, seed=7)
    fresh = simulate_z_trajectories(N_PER_ARM, 10, P, n_sims=40_000, seed=999)
    assert false_positive_rate(fresh, b.z_critical) == pytest.approx(0.05, abs=0.008)


def test_correction_costs_power():
    """Sequential correction is not free -- at fixed n it buys error control
    with power. A tool that hid this would be misleading."""
    p1, p2 = 0.10, 0.115
    n, k = 30_000, 10
    b = calibrate_pocock(n, k, p1, n_sims=40_000)

    corrected = power_under_boundary(n, k, p1, p2, b.z_critical, n_sims=10_000)
    uncorrected_fixed_horizon = power_under_boundary(n, 1, p1, p2, Z_95, n_sims=10_000)

    assert corrected < uncorrected_fixed_horizon
