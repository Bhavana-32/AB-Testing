# How the numbers are verified

A chart looks the same whether the calculation behind it is correct or not, so
the outputs are compared against results that were published and reviewed long
before this project existed.

The test suite validates against **three independent sources**. An error would
have to occur in all three simultaneously to slip through.

← [Back to the README](../README.md)

---

## 1. Against published statistical tables

Armitage, McPherson and Rowe worked out in 1969 how often repeated testing
produces a false positive. The simulation reproduces their table without being
fitted to it:

| Times the results are checked | This simulation | Published (1969) |
| --- | --- | --- |
| 1 | 5.0% | 5.0% |
| 2 | 8.5% | 8.3% |
| 5 | 14.5% | 14.2% |
| 10 | 19.6% | 19.3% |
| 20 | 24.9% | 24.6% |

The same applies to the fix. Pocock published corrected thresholds in 1977.
This project derives them independently by simulation and lands in the same
place:

| Number of checks | Calibrated here | Published (1977) |
| --- | --- | --- |
| 5 | 0.0162 | 0.0158 |
| 10 | 0.0105 | 0.0106 |

See `tests/test_simulation.py`.

---

## 2. Against a second library

Sample sizes are computed two ways: the standard normal approximation used by
most commercial testing tools, and the arcsine transform used by R's `pwr`
package, reached in Python through `statsmodels`.

The tests assert the two agree to within half a percent across realistic
inputs, and separately pin down the specific circumstances where they diverge,
so that divergence is a documented property rather than a surprise.

See `tests/test_calculator.py`.

---

## 3. Against brute force

The strongest check reuses no formulas at all. It generates 20,000 experiments
of simulated visitors at the sample size the calculator recommends, counts how
often the test correctly detects a real effect, and asserts the answer is 80%,
which is what was requested.

If the formula were wrong, this test would fail regardless of how internally
consistent the rest of the code was.

---

## Out of sample validation

Testing a threshold against the same simulations used to derive it would be
circular. The 95th percentile matches by construction, so the check would pass
even if the method were nonsense.

The corrected threshold is therefore calibrated on one batch of simulations and
verified on a completely fresh batch it has never seen. That is the test that
actually establishes the correction works.

---

## A note on a bug this caught

While writing the test suite, a hand transcription of the arcsine sample size
formula dropped a factor of two. The library was right and the hand check was
wrong.

That is the entire argument for writing validation tests before building an
interface. The app would have looked completely normal either way.

---

## Running the suite

```bash
pip install -r requirements-dev.txt
pytest
```

```
58 passed in 2.71s
```

Fast on purpose. A suite slow enough to skip is a suite that gets skipped.
