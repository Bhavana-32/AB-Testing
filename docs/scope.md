# What was deliberately left out

Scope was cut on purpose. Each of these is a real gap, and each was left out
for a stated reason rather than overlooked.

← [Back to the README](../README.md)

---

**Revenue and other continuous metrics.** Working out a sample size for average
revenue per user needs an estimate of how much that figure varies between
users. A planning tool has no way to know this in advance, and revenue
distributions are usually skewed by a small number of large purchases, which
makes the standard approach unreliable. Conversion rates only.

**Bayesian methods.** A legitimate alternative framing rather than a correction
to this one. They answer a different question, and including both would blur
what the project is arguing.

**Always-valid inference and alpha spending.** What large experimentation
platforms actually use, and stronger than Pocock's approach. O'Brien-Fleming
boundaries start very strict and relax over time, which is usually preferred
when stopping early is costly. Implementing one correction thoroughly and
validating it against published values seemed more useful than implementing
three approximately.

**Unequal group sizes.** Sometimes you want 90% of traffic on the safe version.
Real need, straightforward extension, not the first thing most teams require.

**Non-inferiority tests.** Asking "is the new version no worse" rather than "is
it better". Common in practice, different enough in structure to belong in its
own tool.

---

## The limit worth knowing about

Everything here uses a normal approximation to a binomial distribution. That
approximation needs roughly ten expected conversions and ten expected
non-conversions in each group.

Below that the calculator will still return a number, and the number will be
wrong. The app flags this rather than staying quiet, because a confident answer
in a situation the method cannot handle is worse than no answer.

For very rare events, an exact test is the right tool instead.
