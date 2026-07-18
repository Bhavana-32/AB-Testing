# A/B testing, honestly

**Checking an A/B test every morning turns a 5% false positive rate into 20%.
Four out of five "winners" found that way disappear if you keep the test
running.**

Both numbers come from a simulation in this repo, and both match values
published by statisticians in 1969 and 1977.

🚀 **[Try the live app](https://ab-test-validator.streamlit.app/)**

### Early stopping simulation: false positive rate over repeated checks

![Trajectory chart](screenshots/trajectory-chart.png)

### Sample size and experiment duration calculator

![Sample size curve](screenshots/sample-size-curve.png)


---

## What it does

**Checking results early.** Simulates thousands of tests where both groups get
the same thing, so every "winner" is a mistake by construction. Replays one
test day by day, shows how the error rate climbs with each extra check, and
calculates a stricter threshold that fixes it.

**Plan a test.** A sample size and duration calculator that works both ways:
how long do I need, and what can I find in the time I have. Warns you when the
answer is not a usable plan instead of printing a tidy number.

---

## Are the numbers right?

Yes, and that is checked rather than asserted. The simulation reproduces
published tables it was never fitted to:

| Times results are checked | This simulation | Published (1969) |
| --- | --- | --- |
| 1 | 5.0% | 5.0% |
| 10 | 19.6% | 19.3% |
| 20 | 24.9% | 24.6% |

58 tests, running in about 3 seconds, validate against published statistical
tables, a second library, and brute force simulation.

→ [How the numbers are verified](docs/validation.md)

---

## Run it

Python 3.10+. Built on 3.12.

```bash
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

---

## Structure

```
abtest/      The statistics. Pure functions, no Streamlit imports.
views/       The two pages.
tests/       Validation against published values and simulation.
app.py       Navigation and shared styling.
```

Keeping `abtest/` free of interface code is what makes it testable.

---

## More detail

- [The maths in plain language](docs/the-math.md), with a fully worked example
- [How the numbers are verified](docs/validation.md)
- [What was deliberately left out](docs/scope.md), and why
