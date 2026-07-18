# The maths, in plain language

No statistics background assumed. If you can follow long division you can
follow this.

← [Back to the README](../README.md)

---

## Working out a sample size

Everything rests on one tension. Conversion rates bounce around from week to
week even when nothing changes. To claim a real difference, the gap you observe
has to be bigger than the gap noise alone would produce.

Two things decide how much data you need:

1. **How noisy the measurement is.** A conversion rate near 50% is noisier than
   one near 1%, in absolute terms.
2. **How small a difference you are chasing.** This matters far more, because it
   enters the calculation squared. Halving the improvement you want to detect
   roughly quadruples the visitors needed.

The formula:

```
            ( z_confidence × √(2 × p̄ × (1 - p̄))  +  z_power × √(p₁(1-p₁) + p₂(1-p₂)) )²
n per group = ───────────────────────────────────────────────────────────────────────────
                                        (p₂ - p₁)²
```

In words: the top is how much room you need to leave for noise, made up of two
allowances. The first is for calling a winner when there is none. The second is
for the opposite mistake, missing a real improvement. The bottom is the size of
the difference you are hunting. A small difference on the bottom, squared, means
a very large number overall.

---

## A worked example

Your checkout page converts at **5%**. You want to know about anything that
lifts it by **10% or more**, meaning 5% to 5.5%. You want to be **95% sure**
before calling a winner, and to catch a real improvement **80% of the time**.

Filling in the numbers:

- `p₁ = 0.05` and `p₂ = 0.055`, so the difference you are chasing is `0.005`
- `z_confidence = 1.960`, the standard value for being 95% sure
- `z_power = 0.842`, the standard value for catching it 80% of the time
- `p̄ = 0.0525`, the average of the two rates
- `√(2 × 0.0525 × 0.9475) = 0.3154`
- `√(0.05 × 0.95 + 0.055 × 0.945) = 0.3154`

Top of the fraction:

```
(1.960 × 0.3154 + 0.842 × 0.3154)² = 0.8836² = 0.7808
```

Bottom of the fraction:

```
0.005² = 0.000025
```

Divide:

```
0.7808 ÷ 0.000025 = 31,233
```

So you need **31,234 visitors per version**, rounding up, or **62,468 in
total**. At 2,000 visitors a day that is 31.2 days, which the tool rounds up to
**35 days**, five whole weeks.

### Two details in that last step

**Whole weeks, not calendar days.** Shopping behaviour differs by day of the
week. A test running 31 days contains five Mondays but only four Saturdays,
which tilts the result. Whole weeks give each day equal weight.

**Sensitivity to the target.** If a 5% improvement was worth knowing about
instead of 10%, the requirement jumps from 31,234 to **122,124** visitors per
version. Same page, same traffic, four times the wait. This is why "let's just
see if it does anything" is an expensive request.

---

## Why checking early breaks it

That calculation assumes you look **once**, at a moment fixed before the test
started.

Every additional look is another opportunity for random noise to cross the
threshold. Early in a test there is very little data, the two rates swing
widely, and sometimes they swing far apart for no reason at all. If your rule is
"stop as soon as it looks significant", you are not running one test at 95%
confidence. You are running twenty and keeping whichever looks best.

Hence 5% becoming 20%, and hence roughly 80% of those apparent winners
evaporating by the end of the test.

---

## The fix

Pocock's correction keeps a single threshold and applies it at every check, but
raises it so the chance of *ever* crossing it comes back to 5%. For ten checks
that means requiring roughly 98.95% confidence at each look rather than 95%.

This project derives that threshold by simulation rather than looking it up. The
logic is simple: the correct threshold is, by definition, the point that only 5%
of no-effect experiments ever exceed. Simulate a large batch of no-effect
experiments, take the 95th percentile of the largest test statistic each one
reaches, and that is the threshold. No table needed, and the answer can be
checked against Pocock's published values afterwards.

**The correction is not free.** A higher bar means real improvements get missed
more often. At a fixed sample size, one setup in the app trades 94% detection
down to 85%. Any tool showing you the fix without the cost is leaving something
out.

---

## Two ways of doing the same sum

Sample sizes here can be computed two ways:

- **Standard**, the normal approximation used by most commercial testing tools.
- **Arcsine**, a variance-stabilising transform, used by R's `pwr` package and
  available in Python through `statsmodels`.

For realistic tests the two agree to within a fraction of a percent. They only
pull apart for very large relative improvements, several times the starting
rate, which is rare in practice. The app shows both so the choice is visible
rather than hidden.

---

## References

- Armitage, P., McPherson, C. K., and Rowe, B. C. (1969). Repeated significance
  tests on accumulating data. *Journal of the Royal Statistical Society, Series
  A*, 132(2), 235 to 244.
- Pocock, S. J. (1977). Group sequential methods in the design and analysis of
  clinical trials. *Biometrika*, 64(2), 191 to 199.
- Johari, R., Koomen, P., Pekelis, L., and Walsh, D. (2017). Peeking at A/B
  tests: why it matters, and what to do about it. *KDD 2017*.
- Kohavi, R., Tang, D., and Xu, Y. (2020). *Trustworthy Online Controlled
  Experiments*. Cambridge University Press.
