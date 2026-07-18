"""
Monte Carlo simulation of the "peeking problem" in A/B testing.

The setup is an A/A test: both arms are drawn from the *same* true conversion
rate, so there is no real effect. Every "significant" result is therefore a
false positive by construction, and the interpretation of the output is
unambiguous.

If you run a fixed-horizon test and look once at the end, the false positive
rate is alpha (5%). If you look repeatedly and stop the first time you see
p < 0.05, the false positive rate is much higher, because you get a fresh
chance to cross the threshold at every look.

Published ground truth (Armitage, McPherson & Rowe 1969, "Repeated
significance tests on accumulating data", JRSS-A 132(2)) for two-sided
alpha = 0.05 -- ARMITAGE_1969 below. The simulation is validated against it.

Everything is vectorised over simulations. A naive Python loop over 10,000
simulations x 20 looks takes long enough that the Streamlit app would feel
broken; the array version runs in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# Probability of at least one p < 0.05 (two-sided) across K equally spaced
# looks at accumulating data, under the null. Armitage et al. (1969).
ARMITAGE_1969: dict[int, float] = {
    1: 0.050,
    2: 0.083,
    3: 0.107,
    4: 0.126,
    5: 0.142,
    10: 0.193,
    20: 0.246,
}

# Pocock (1977) constant nominal significance levels that hold the overall
# two-sided false positive rate at 0.05 across K equally spaced looks.
POCOCK_NOMINAL_ALPHA: dict[int, float] = {
    1: 0.0500,
    2: 0.0294,
    3: 0.0221,
    4: 0.0182,
    5: 0.0158,
    10: 0.0106,
    20: 0.0075,
}


# --------------------------------------------------------------------------
# Core simulation
# --------------------------------------------------------------------------


@dataclass
class ExperimentPaths:
    """Everything about a batch of simulated experiments, look by look.

    `cum_n` has shape (n_looks,); the conversion count arrays and `z` have
    shape (n_sims, n_looks).
    """

    cum_n: np.ndarray
    conversions_control: np.ndarray
    conversions_treatment: np.ndarray
    z: np.ndarray

    @property
    def rate_control(self) -> np.ndarray:
        return self.conversions_control / self.cum_n

    @property
    def rate_treatment(self) -> np.ndarray:
        return self.conversions_treatment / self.cum_n


def simulate_experiments(
    n_per_arm: int,
    n_looks: int,
    p_control: float,
    p_treatment: float | None = None,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> ExperimentPaths:
    """Simulate `n_sims` experiments, keeping the raw conversion counts.

    Users arrive gradually; the test is evaluated at `n_looks` equally spaced
    points, the last of which uses the full `n_per_arm` per arm.

    `p_treatment=None` gives an A/A test (no true effect), which is the
    configuration used to measure false positive rates.

    The counts are retained (rather than only the z-statistics) so the app can
    replay a single experiment the way a team would actually have seen it:
    two conversion rates and a verdict, day by day.
    """
    if n_looks < 1:
        raise ValueError("n_looks must be at least 1")
    if n_per_arm < n_looks:
        raise ValueError("n_per_arm must be at least n_looks (one user per look)")
    if p_treatment is None:
        p_treatment = p_control

    rng = np.random.default_rng(seed)

    # Cumulative sample size at each look, equally spaced, ending at n_per_arm.
    cum_n = np.round(np.linspace(n_per_arm / n_looks, n_per_arm, n_looks)).astype(np.int64)
    cum_n = np.maximum.accumulate(np.maximum(cum_n, 1))
    increments = np.diff(cum_n, prepend=0)

    # Draw conversions in each interval, then accumulate. Simulating the
    # increments (rather than redrawing each look) is what makes the looks
    # correctly correlated -- data accumulates, it is not resampled.
    conv_c = rng.binomial(increments, p_control, size=(n_sims, n_looks)).cumsum(axis=1)
    conv_t = rng.binomial(increments, p_treatment, size=(n_sims, n_looks)).cumsum(axis=1)

    return ExperimentPaths(
        cum_n=cum_n,
        conversions_control=conv_c,
        conversions_treatment=conv_t,
        z=_pooled_z(conv_c, conv_t, cum_n),
    )


def simulate_z_trajectories(
    n_per_arm: int,
    n_looks: int,
    p_control: float,
    p_treatment: float | None = None,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> np.ndarray:
    """Just the z-statistics from `simulate_experiments`, shape (n_sims, n_looks)."""
    return simulate_experiments(
        n_per_arm=n_per_arm,
        n_looks=n_looks,
        p_control=p_control,
        p_treatment=p_treatment,
        n_sims=n_sims,
        seed=seed,
    ).z


def _pooled_z(conv_c: np.ndarray, conv_t: np.ndarray, cum_n: np.ndarray) -> np.ndarray:
    """Two-proportion pooled z-test, vectorised over sims and looks."""
    n = cum_n.astype(float)
    p_c = conv_c / n
    p_t = conv_t / n
    p_pool = (conv_c + conv_t) / (2.0 * n)
    var = p_pool * (1.0 - p_pool) * (2.0 / n)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (p_t - p_c) / np.sqrt(var)
    # Zero conversions in both arms -> zero variance -> undefined z.
    # No evidence of a difference, so treat as z = 0 rather than propagating NaN.
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
# The peeking problem
# --------------------------------------------------------------------------


@dataclass
class PeekingResult:
    n_looks: int
    false_positive_rate: float
    fixed_horizon_rate: float
    z_critical: float
    published_reference: float | None = None


def false_positive_rate(
    z: np.ndarray, z_critical: float, alternative: str = "two-sided"
) -> float:
    """Fraction of simulated experiments that cross the boundary at ANY look.

    This is the "stop as soon as it's significant" decision rule.
    """
    stat = np.abs(z) if alternative == "two-sided" else z
    return float((stat.max(axis=1) > z_critical).mean())


def peeking_curve(
    n_per_arm: int,
    p_control: float,
    look_counts: tuple[int, ...] = (1, 2, 3, 4, 5, 10, 20),
    alpha: float = 0.05,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> list[PeekingResult]:
    """False positive rate as a function of how many times you peek.

    The headline result of the project. Compare against ARMITAGE_1969.
    """
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    results = []
    for i, k in enumerate(look_counts):
        z = simulate_z_trajectories(
            n_per_arm=n_per_arm,
            n_looks=k,
            p_control=p_control,
            n_sims=n_sims,
            seed=None if seed is None else seed + i,
        )
        results.append(
            PeekingResult(
                n_looks=k,
                false_positive_rate=false_positive_rate(z, z_crit),
                fixed_horizon_rate=alpha,
                z_critical=z_crit,
                published_reference=ARMITAGE_1969.get(k),
            )
        )
    return results


def p_values(z: np.ndarray) -> np.ndarray:
    """Two-sided p-values from z-statistics, elementwise."""
    return 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))


def first_crossing(z: np.ndarray, z_critical: float) -> np.ndarray:
    """Index of the first look at which each simulation crosses; -1 if never.

    Used to show *when* false positives happen -- disproportionately early,
    when the estimate is noisiest, which is exactly when a team is most
    tempted to call the test.
    """
    crossed = np.abs(z) > z_critical
    any_crossed = crossed.any(axis=1)
    idx = crossed.argmax(axis=1)
    return np.where(any_crossed, idx, -1)


def pick_false_positive_example(z: np.ndarray, z_critical: float) -> int | None:
    """Index of one experiment that looked like a winner early and was not.

    Chooses among experiments that cross the threshold at some point but end
    non-significant, preferring the one that crossed earliest. Returns None if
    no such experiment exists in the batch.

    Used to put a face on the aggregate statistic: a single test a team would
    have called, shipped, and been wrong about.
    """
    crossed = np.abs(z) > z_critical
    ever = crossed.any(axis=1)
    ends_significant = np.abs(z[:, -1]) > z_critical
    candidates = np.where(ever & ~ends_significant)[0]
    if len(candidates) == 0:
        return None
    first_look = crossed[candidates].argmax(axis=1)
    return int(candidates[first_look.argmin()])


def recovery_rate(z: np.ndarray, z_critical: float) -> float:
    """Fraction of experiments that cross the boundary but end non-significant.

    These are the tests a team would have "won" by stopping early, and lost
    by waiting. The most persuasive single number in the whole simulation.
    """
    crossed_ever = (np.abs(z) > z_critical).any(axis=1)
    significant_at_end = np.abs(z[:, -1]) > z_critical
    n_crossed = crossed_ever.sum()
    if n_crossed == 0:
        return 0.0
    return float((crossed_ever & ~significant_at_end).sum() / n_crossed)


# --------------------------------------------------------------------------
# The fix: Pocock's group sequential boundary, calibrated by simulation
# --------------------------------------------------------------------------


@dataclass
class PocockBoundary:
    n_looks: int
    z_critical: float
    nominal_alpha: float
    target_alpha: float
    published_nominal_alpha: float | None = None
    meta: dict = field(default_factory=dict)


def calibrate_pocock(
    n_per_arm: int,
    n_looks: int,
    p_control: float,
    alpha: float = 0.05,
    n_sims: int = 100_000,
    seed: int | None = 42,
) -> PocockBoundary:
    """Find the constant z boundary that holds overall false positives at alpha.

    Pocock's boundary uses the *same* critical value at every look, chosen so
    that the probability of crossing it at least once equals alpha. That
    quantity is, by definition, the (1 - alpha) quantile of the distribution
    of max|z| across looks under the null -- so no root-finding or iterative
    search is needed. Simulate the null once, take the quantile, done.

    The result is checked against Pocock's published table in the test suite.
    """
    z = simulate_z_trajectories(
        n_per_arm=n_per_arm,
        n_looks=n_looks,
        p_control=p_control,
        n_sims=n_sims,
        seed=seed,
    )
    max_abs_z = np.abs(z).max(axis=1)
    z_crit = float(np.quantile(max_abs_z, 1.0 - alpha))
    nominal = float(2.0 * (1.0 - stats.norm.cdf(z_crit)))

    return PocockBoundary(
        n_looks=n_looks,
        z_critical=z_crit,
        nominal_alpha=nominal,
        target_alpha=alpha,
        published_nominal_alpha=POCOCK_NOMINAL_ALPHA.get(n_looks),
        meta={"n_sims": n_sims, "n_per_arm": n_per_arm, "p_control": p_control},
    )


def power_under_boundary(
    n_per_arm: int,
    n_looks: int,
    p_control: float,
    p_treatment: float,
    z_critical: float,
    n_sims: int = 10_000,
    seed: int | None = 7,
) -> float:
    """Power of a sequential test that stops at the first boundary crossing.

    Included because corrections are not free: raising the boundary to control
    false positives costs power at a fixed sample size. A tool that shows the
    fix without its cost is only telling half the story.
    """
    z = simulate_z_trajectories(
        n_per_arm=n_per_arm,
        n_looks=n_looks,
        p_control=p_control,
        p_treatment=p_treatment,
        n_sims=n_sims,
        seed=seed,
    )
    return float((np.abs(z).max(axis=1) > z_critical).mean())
