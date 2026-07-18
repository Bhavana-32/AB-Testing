"""
The peeking problem, told as a product story.

Deliberately the first page a visitor sees. The sample size calculator is
context; this is the argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from abtest.simulation import (
    ARMITAGE_1969,
    calibrate_pocock,
    false_positive_rate,
    first_crossing,
    p_values,
    peeking_curve,
    pick_false_positive_example,
    power_under_boundary,
    recovery_rate,
    simulate_experiments,
)

INK = "#16181D"
MUTED = "#69737F"
GREY = "#CBD2D9"
RED = "#E04E4E"
BLUE = "#3B6FD4"
GREEN = "#149C8E"
SAND = "#F7F8FA"


# --------------------------------------------------------------------------
# Cached compute. Community Cloud is slow, so nothing recomputes unless an
# input actually changes.
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def run_experiments(n_per_arm, n_checks, p, n_sims, seed):
    return simulate_experiments(
        n_per_arm=n_per_arm, n_looks=n_checks, p_control=p, n_sims=n_sims, seed=seed
    )


@st.cache_data(show_spinner=False)
def run_peeking_curve(n_per_arm, p, check_counts, alpha, n_sims):
    return peeking_curve(
        n_per_arm=n_per_arm,
        p_control=p,
        look_counts=check_counts,
        alpha=alpha,
        n_sims=n_sims,
    )


@st.cache_data(show_spinner=False)
def run_pocock(n_per_arm, n_checks, p, alpha, n_sims):
    return calibrate_pocock(
        n_per_arm=n_per_arm, n_looks=n_checks, p_control=p, alpha=alpha, n_sims=n_sims
    )


# --------------------------------------------------------------------------
# Your experiment
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Your experiment")
    st.caption(
        "Set this up the way your team actually runs tests. "
        "Everything on the page updates."
    )

    p_control = st.slider(
        "Share of visitors who buy today",
        0.01,
        0.40,
        0.10,
        0.01,
        format="%.0f%%",
        help="Your current conversion rate, before you change anything.",
    )
    daily_visitors = st.select_slider(
        "Visitors per day in each group",
        options=[200, 500, 1_000, 2_000, 5_000],
        value=500,
    )
    test_days = st.slider("How many days the test runs", 7, 42, 21)
    check_every = st.select_slider(
        "How often someone opens the dashboard",
        options=[1, 2, 3, 7, 21],
        value=1,
        format_func=lambda d: {
            1: "Every day",
            2: "Every 2 days",
            3: "Every 3 days",
            7: "Once a week",
            21: "Only at the very end",
        }[d],
    )

    with st.expander("Fine print"):
        alpha = st.select_slider(
            "How sure you want to be before calling a winner",
            options=[0.01, 0.05, 0.10],
            value=0.05,
            format_func=lambda a: f"{1 - a:.0%} sure",
        )
        n_sims = st.select_slider(
            "How many times to repeat the whole experiment",
            options=[2_000, 10_000, 20_000, 50_000],
            value=10_000,
            help="More repeats give steadier numbers and a slower page.",
        )

n_checks = max(1, test_days // check_every)
n_per_arm = daily_visitors * test_days

z_naive = float(stats.norm.ppf(1 - alpha / 2))
paths = run_experiments(n_per_arm, n_checks, p_control, n_sims, 1)
z = paths.z
pvals = p_values(z)
fpr = false_positive_rate(z, z_naive)
recov = recovery_rate(z, z_naive)
days_axis = paths.cum_n / daily_visitors


# --------------------------------------------------------------------------
# The story
# --------------------------------------------------------------------------

st.markdown('<p class="kicker">Product experimentation</p>', unsafe_allow_html=True)
st.title("What happens when you check results early")
st.markdown(
    f'<p class="lede">We ran the same experiment {n_sims:,} times with nothing '
    f"to find. The dashboard announced a winner in <b>{fpr:.0%}</b> of them. "
    "This page shows how that happens, and what to do about it.</p>",
    unsafe_allow_html=True,
)

st.write("")

st.markdown(
    f"""
    <div class="scenario">
    Your team changes the checkout button. Half your visitors see the old one,
    half see the new one, and you wait {test_days} days.<br><br>
    Except nobody really waits. Someone opens the dashboard on Monday, again on
    Tuesday, and by Thursday the tool says the new button is ahead. You ship it.
    <br><br>
    <b>So here is the test.</b> We are going to run that same experiment
    {n_sims:,} times, except we secretly give both groups the <i>identical</i>
    button. Nothing changed. There is no winner to find. Then we count how often
    the dashboard announces one anyway.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.write("")

c1, c2, c3 = st.columns(3)
c1.metric(
    "Tests where the dashboard called a winner",
    f"{fpr:.1%}",
    help="Remember: there was no winner. Every one of these is wrong.",
)
c2.metric(
    "How often that should happen",
    f"{alpha:.0%}",
    help="The error rate you agreed to when you picked your confidence level.",
)
c3.metric(
    "Winners that vanished by the last day",
    f"{recov:.0%}",
    help=(
        "Of the tests that looked like winners at some point, this share were "
        "back to no difference by the end."
    ),
)

st.write("")
if n_checks > 1:
    st.markdown(
        f'<p class="punch">Checking {n_checks} times turned a {alpha:.0%} error rate '
        f"into {fpr:.0%}. That is {fpr / alpha:.1f} times more wrong calls than you "
        "signed up for, on experiments where nothing was happening at all.</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<p class="punch">Looking once, at the end, gives you exactly the {alpha:.0%} '
        "error rate you asked for. This is the honest version. "
        "Now go back and check more often.</p>",
        unsafe_allow_html=True,
    )

st.divider()


# --------------------------------------------------------------------------
# One team's experiment, day by day
# --------------------------------------------------------------------------

st.header("What this feels like from the inside")

example = pick_false_positive_example(z, z_naive)

if example is None:
    st.success(
        "In this setup, not one of the simulated tests produced a false winner "
        "that later disappeared. Try checking more often, or lowering how sure "
        "you want to be."
    )
else:
    cross_look = int((np.abs(z[example]) > z_naive).argmax())
    cross_day = int(round(days_axis[cross_look]))

    rate_c = paths.conversions_control[example] / paths.cum_n
    rate_t = paths.conversions_treatment[example] / paths.cum_n
    lift = np.where(rate_c > 0, (rate_t - rate_c) / np.where(rate_c > 0, rate_c, 1), 0.0)
    verdicts = np.where(
        np.abs(z[example]) > z_naive, "New button wins", "No clear difference"
    )

    st.markdown(
        f"""
Here is one of those tests, exactly as the team would have seen it. Both groups
are seeing the **same button**. Any difference below is noise.
"""
    )

    a, b = st.columns([1, 1])
    a.markdown(
        f"""
        <div class="scenario" style="border-left-color:{RED};">
        <b>Day {cross_day}.</b> The new button is at
        {rate_t[cross_look]:.2%} against {rate_c[cross_look]:.2%} for the old one.
        That is a {lift[cross_look]:+.0%} lift, and the tool marks it as a win.
        A reasonable person ships this and writes it up.
        </div>
        """,
        unsafe_allow_html=True,
    )
    b.markdown(
        f"""
        <div class="scenario" style="border-left-color:{GREEN};">
        <b>Day {test_days}.</b> The two buttons are at
        {rate_t[-1]:.2%} and {rate_c[-1]:.2%}. The gap is
        {lift[-1]:+.0%} and the tool no longer sees anything.
        The win on day {cross_day} was never there.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    table = pd.DataFrame(
        {
            "Day": np.round(days_axis).astype(int),
            "Visitors per group": paths.cum_n,
            "Old button": [f"{r:.2%}" for r in rate_c],
            "New button": [f"{r:.2%}" for r in rate_t],
            "Difference": [f"{l:+.1%}" for l in lift],
            "What the dashboard says": verdicts,
        }
    )

    def highlight(row):
        if row["What the dashboard says"] == "New button wins":
            return ["background-color: #FDECEC"] * len(row)
        return [""] * len(row)

    st.dataframe(
        table.style.apply(highlight, axis=1),
        hide_index=True,
        width="stretch",
        height=min(420, 40 + 35 * len(table)),
    )
    st.caption(
        "Red rows are days the team would have declared victory. "
        "Nothing about the two buttons was ever different."
    )

st.divider()


# --------------------------------------------------------------------------
# Many experiments at once
# --------------------------------------------------------------------------

st.header("Now watch sixty teams do it at the same time")

st.markdown(
    '<p class="lede">Each line is one experiment running over time. The line '
    "shows how convinced the tool is that it has found a difference. When a "
    "line dips below the dotted line, the dashboard is calling a winner.</p>",
    unsafe_allow_html=True,
)

n_show = 60
crossed = (np.abs(z[:n_show]) > z_naive).any(axis=1)
ends_significant = np.abs(z[:n_show, -1]) > z_naive
false_alarm = crossed & ~ends_significant

fig = go.Figure()
for i in range(n_show):
    if false_alarm[i]:
        continue
    fig.add_trace(
        go.Scatter(
            x=days_axis,
            y=pvals[i],
            mode="lines",
            line=dict(color=GREY, width=1),
            opacity=0.55,
            hoverinfo="skip",
            showlegend=False,
        )
    )

fa_idx = np.where(false_alarm)[0]
for j, i in enumerate(fa_idx):
    fig.add_trace(
        go.Scatter(
            x=days_axis,
            y=pvals[i],
            mode="lines",
            line=dict(color=RED, width=2),
            name="Called a winner, then lost it",
            legendgroup="fa",
            showlegend=bool(j == 0),
            hovertemplate="Day %{x:.0f}<br>%{y:.3f}<extra></extra>",
        )
    )

fig.add_hrect(
    y0=1e-4, y1=alpha, fillcolor=RED, opacity=0.05, line_width=0, layer="below"
)
fig.add_hline(
    y=alpha,
    line=dict(color=INK, width=1.5, dash="dot"),
    annotation_text="  dashboard calls a winner below this line",
    annotation_position="bottom left",
    annotation_font=dict(size=12, color=MUTED),
)
fig.update_layout(
    height=440,
    xaxis_title="Day of the test",
    yaxis_title="How convinced the tool is (lower means more convinced)",
    yaxis_type="log",
    yaxis_range=[-3, 0],
    yaxis_showticklabels=False,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    plot_bgcolor="white",
    hovermode="closest",
)
fig.update_xaxes(showgrid=False, linecolor=GREY)
fig.update_yaxes(gridcolor="#EEF1F4")
st.plotly_chart(fig, width="stretch")

st.caption(
    f"{n_show} of the {n_sims:,} simulated tests. Grey lines never got called. "
    "Red lines did, and then went back to nothing."
)

idx = first_crossing(z, z_naive)
crossings = idx[idx >= 0]
if len(crossings) and n_checks > 2:
    early = float((crossings < n_checks / 2).mean())
    st.info(
        f"**{early:.0%} of the false winners show up in the first half of the test.** "
        "Early on there is very little data, so the numbers swing around a lot. "
        "That is also when everyone is most excited to look."
    )

st.divider()


# --------------------------------------------------------------------------
# The curve
# --------------------------------------------------------------------------

st.header("Every extra check makes it worse")

check_counts = (1, 2, 3, 4, 5, 10, 20)
curve = run_peeking_curve(n_per_arm, p_control, check_counts, alpha, n_sims)
sim_rates = [r.false_positive_rate for r in curve]

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=list(check_counts),
        y=sim_rates,
        mode="lines+markers",
        name="This simulation",
        line=dict(color=RED, width=3),
        marker=dict(size=9),
        hovertemplate="Checked %{x} times<br>%{y:.1%} of tests wrongly called a winner<extra></extra>",
    )
)
if abs(alpha - 0.05) < 1e-9:
    fig2.add_trace(
        go.Scatter(
            x=list(ARMITAGE_1969.keys()),
            y=list(ARMITAGE_1969.values()),
            mode="markers",
            name="Published values from a 1969 statistics paper",
            marker=dict(size=14, symbol="circle-open", color=BLUE, line=dict(width=2.5)),
            hovertemplate="Checked %{x} times<br>%{y:.1%} published<extra></extra>",
        )
    )
fig2.add_hline(
    y=alpha,
    line=dict(color=GREEN, width=2, dash="dash"),
    annotation_text=f"  what you think you are getting ({alpha:.0%})",
    annotation_font=dict(size=12, color=GREEN),
)
fig2.update_layout(
    height=400,
    xaxis_title="Number of times someone checks the results",
    yaxis_title="Share of tests wrongly called a winner",
    yaxis_tickformat=".0%",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    plot_bgcolor="white",
)
fig2.update_xaxes(showgrid=False, linecolor=GREY)
fig2.update_yaxes(gridcolor="#EEF1F4")
st.plotly_chart(fig2, width="stretch")

if abs(alpha - 0.05) < 1e-9:
    st.caption(
        "The blue circles are numbers published by three statisticians in 1969, "
        "long before anyone had an A/B testing dashboard. This simulation was "
        "not tuned to match them. It lands on them on its own, which is how you "
        "know the simulation is doing the right thing."
    )

st.divider()


# --------------------------------------------------------------------------
# The fix
# --------------------------------------------------------------------------

st.header("You can keep checking. You just need a higher bar.")

st.markdown(
    '<p class="lede">The problem is not that people look at the dashboard. '
    "The problem is looking every day while using a standard built for looking "
    "once. Raise the bar for every check, and the error rate comes back to "
    "where it should be.</p>",
    unsafe_allow_html=True,
)

st.write("")

if n_checks >= 2:
    with st.spinner("Working out the right bar for your setup..."):
        boundary = run_pocock(n_per_arm, n_checks, p_control, alpha, max(n_sims, 20_000))

    fresh = run_experiments(n_per_arm, n_checks, p_control, n_sims, 999).z
    fpr_corrected = false_positive_rate(fresh, boundary.z_critical)

    b1, b2, b3 = st.columns(3)
    b1.metric(
        "Your new bar",
        f"{1 - boundary.nominal_alpha:.2%} sure",
        help=(
            f"Instead of {1 - alpha:.0%} sure. You need to be this confident at "
            f"every one of your {n_checks} checks before calling a winner."
        ),
    )
    b2.metric("Wrong calls before", f"{fpr:.0%}")
    b3.metric("Wrong calls after", f"{fpr_corrected:.0%}", delta="back where it should be", delta_color="off")

    st.caption(
        "This bar was worked out by simulation, and it matches the value "
        "published by the statistician Stuart Pocock in 1977. The corrected "
        "number above is measured on a completely fresh batch of simulated "
        "tests, not the ones used to set the bar."
    )

    st.write("")
    st.markdown("##### There is a catch")

    lift = st.slider(
        "Suppose the new button really were better by this much",
        0.02,
        0.30,
        0.15,
        0.01,
        format="%.0f%%",
    )
    p_treat = min(p_control * (1 + lift), 0.999)
    pw_corrected = power_under_boundary(
        n_per_arm, n_checks, p_control, p_treat, boundary.z_critical, n_sims=5_000
    )
    pw_fixed = power_under_boundary(
        n_per_arm, 1, p_control, p_treat, z_naive, n_sims=5_000
    )

    d1, d2 = st.columns(2)
    d1.metric("Chance you catch it, checking once at the end", f"{pw_fixed:.0%}")
    d2.metric(
        "Chance you catch it, checking daily with the higher bar",
        f"{pw_corrected:.0%}",
        delta=f"{pw_corrected - pw_fixed:+.0%}",
    )
    st.caption(
        "A higher bar means fewer false winners, but it also means you miss some "
        "real ones. Nothing here is free. Anyone selling you a fix without a cost "
        "is leaving something out."
    )
else:
    st.info(
        "You are only checking once, so there is nothing to correct. "
        "Set the dashboard to be checked more often to see the fix."
    )

st.divider()


# --------------------------------------------------------------------------
# Plain language notes
# --------------------------------------------------------------------------

col_a, col_b = st.columns(2)

with col_a:
    with st.expander("What the words on this page actually mean"):
        st.markdown(
            """
**Conversion rate.** Out of everyone who saw the page, the share who did the
thing you wanted. Bought something, signed up, clicked the button.

**Calling a winner.** Your testing tool decides the difference between the two
groups is big enough that it probably is not just luck. Every tool has some
version of this, usually a green tick or the word "significant".

**How sure you want to be.** Before you start, you pick how much risk of being
wrong you will accept. Most teams use 95% sure, which means you accept being
wrong about 1 test in 20. That last part is the bit people forget.

**A false winner.** The tool says one version is better when really they are
the same. You ship a change that does nothing, and you believe it worked.

**Checking, or peeking.** Looking at the results before the test has finished,
and being willing to stop early if you like what you see. Everyone does it.
That is the whole point of this page.
"""
        )

with col_b:
    with st.expander("Why this happens, in one paragraph"):
        st.markdown(
            """
Being "95% sure" only means what you think it means if you look **once**, at a
moment you picked before you started.

Every extra look is another roll of the dice. Early in a test you have very
little data, so the two numbers bounce around a lot, and sometimes they bounce
far apart purely by chance. If you have agreed with yourself to stop the moment
the tool says "winner", then you are not running one test at 95% confidence.
You are running twenty of them and keeping whichever one happens to look good.

The fix is not to stop looking. It is to be harder to convince, because you are
giving yourself more chances to be convinced.
"""
        )

with st.expander("What large companies do about this"):
    st.markdown(
        """
The correction on this page is the simplest one that works, which is why it is
the one built here. Companies running thousands of tests a year go further:

- **A bar that starts very high and eases off.** Early results have to be
  extraordinary to stop a test, and the standard relaxes as data comes in. This
  is usually preferred when stopping early is expensive.
- **Results that are safe to read at any moment.** Some platforms, including
  Optimizely, use methods designed from the ground up for continuous watching,
  so there is no wrong time to look.
- **A different framework entirely.** Some teams drop the "is it significant"
  question and instead ask how much money a decision is likely to cost or make.

All of them trade something away for the freedom to look whenever you like.
None of them make looking free.
"""
    )

st.write("")
st.caption(
    "Simulation checked against: Armitage, McPherson and Rowe (1969), Journal of "
    "the Royal Statistical Society A, 132(2). Pocock (1977), Biometrika, 64(2). "
    "Further reading: Johari, Koomen, Pekelis and Walsh, Peeking at A/B Tests, KDD 2017."
)
