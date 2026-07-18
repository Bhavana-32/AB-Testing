"""
Plan a test: how many visitors, and how many days.

The companion to the peeking page. That page argues you should not stop a test
early. This one answers the obvious next question, which is how long you are
being asked to wait and why.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

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

INK = "#16181D"
MUTED = "#69737F"
GREY = "#CBD2D9"
RED = "#E04E4E"
BLUE = "#3B6FD4"
GREEN = "#149C8E"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Your test")

    baseline = st.slider(
        "Share of visitors who convert today",
        0.005,
        0.50,
        0.10,
        0.005,
        format="%.1f%%",
        help="Your current rate, before you change anything.",
    )
    lift = st.slider(
        "Smallest improvement worth finding",
        0.01,
        0.50,
        0.10,
        0.01,
        format="%.0f%%",
        help=(
            "A 10% improvement on a 5% conversion rate means going from 5% to "
            "5.5%, not from 5% to 15%."
        ),
    )
    daily_visitors = st.number_input(
        "Visitors you can send to the test each day",
        min_value=50,
        max_value=1_000_000,
        value=2_000,
        step=100,
        help="Across all versions combined, not per version.",
    )
    n_variants = st.select_slider(
        "How many versions you are testing",
        options=[2, 3, 4, 5],
        value=2,
        format_func=lambda k: f"{k} versions (1 original, {k - 1} new)",
    )

    with st.expander("Fine print"):
        confidence = st.select_slider(
            "How sure you want to be before calling a winner",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda c: f"{c:.0%} sure",
        )
        power = st.select_slider(
            "How often you want to catch a real improvement",
            options=[0.70, 0.80, 0.90],
            value=0.80,
            format_func=lambda p: f"{p:.0%} of the time",
        )
        allocation = st.slider(
            "Share of your traffic entering the test",
            0.1,
            1.0,
            1.0,
            0.1,
            format="%.0f%%",
        )
        method = st.radio(
            "Calculation method",
            options=["pooled", "arcsine"],
            format_func=lambda m: {
                "pooled": "Standard (matches most testing tools)",
                "arcsine": "Alternative (matches R's pwr package)",
            }[m],
            help=(
                "Two accepted ways of doing the same sum. They agree closely "
                "for realistic tests. Both are shown further down."
            ),
        )

alpha = 1 - confidence
adjusted_alpha = bonferroni_alpha(alpha, n_variants)
target = treatment_rate(baseline, lift, "relative")


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown('<p class="kicker">Before you press start</p>', unsafe_allow_html=True)
st.title("How long will this take?")
st.markdown(
    '<p class="lede">Nearly every argument about an A/B test is really an '
    "argument about patience. This works out how much you need, before you "
    "start, so it is a plan rather than a negotiation.</p>",
    unsafe_allow_html=True,
)

st.write("")

tab_plan, tab_reverse = st.tabs(
    ["  How long do I need?  ", "  What can I find in the time I have?  "]
)


# --------------------------------------------------------------------------
# Forward: how much do I need?
# --------------------------------------------------------------------------

with tab_plan:
    n_needed = sample_size(
        baseline, target, alpha=adjusted_alpha, power=power, method=method
    )
    plan = duration(
        n_per_group=n_needed,
        daily_traffic=daily_visitors,
        n_groups=n_variants,
        allocation=allocation,
        round_to_whole_weeks=True,
    )

    st.markdown(
        f"""
        <div class="scenario">
        You want to spot a move from <b>{baseline:.2%}</b> to
        <b>{target:.2%}</b>, which is the {lift:.0%} improvement you asked for.
        To be {confidence:.0%} sure and to catch it {power:.0%} of the time,
        here is what that costs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    m1, m2, m3 = st.columns(3)
    m1.metric("Visitors needed per version", f"{n_needed:,}")
    m2.metric("Visitors needed in total", f"{plan.total_sample_needed:,}")
    m3.metric(
        "Days to run",
        f"{plan.days}",
        delta=f"{plan.weeks:.0f} weeks",
        delta_color="off",
    )

    st.caption(plan.note + f" Without rounding it would be {plan.raw_days:.1f} days.")

    warnings = normal_approximation_warnings(baseline, target, n_needed)
    if warnings:
        st.warning(
            "**Treat these numbers carefully.** "
            + " ".join(warnings)
            + " In plain terms: at conversion rates this low, so few people "
            "convert that the usual maths stops being trustworthy."
        )

    if plan.days > 180:
        st.error(
            f"**{plan.days} days is not a real plan.** A test running over six "
            "months will be overtaken by other changes to the product, seasonal "
            "swings, and shifts in who is visiting. Either look for a bigger "
            "improvement, send more traffic to the test, or accept that this "
            "particular question cannot be answered by an A/B test right now."
        )
    elif plan.days > 56:
        st.warning(
            f"**{plan.days} days is a long time.** Worth checking whether the "
            "improvement you are looking for is really the smallest one that "
            "would change your decision. Being slightly less ambitious here is "
            "usually cheaper than waiting two months."
        )

    if n_variants > 2:
        n_two = sample_size(baseline, target, alpha=alpha, power=power, method=method)
        st.info(
            f"**Testing {n_variants} versions costs you extra.** Comparing more "
            "things means more chances to be fooled by luck, so the bar for each "
            f"comparison has to rise. Two versions would need {n_two:,} visitors "
            f"each instead of {n_needed:,}."
        )

    st.write("")
    st.subheader("Why small improvements are so expensive")

    st.markdown(
        '<p class="lede">This is the chart to bring to the meeting where '
        'somebody says "can we just see if it does anything".</p>',
        unsafe_allow_html=True,
    )

    lifts = np.linspace(0.01, 0.50, 60)
    curve = sample_size_curve(
        baseline, lifts, "relative", alpha=adjusted_alpha, power=power, method=method
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lifts,
            y=curve,
            mode="lines",
            line=dict(color=BLUE, width=3),
            name="Visitors needed",
            hovertemplate="To find a %{x:.0%} improvement<br>%{y:,.0f} visitors per version<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lift],
            y=[n_needed],
            mode="markers+text",
            marker=dict(size=15, color=RED),
            text=[f"  you are here: {n_needed:,}"],
            textposition="middle right",
            textfont=dict(size=13, color=RED),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=430,
        xaxis_title="Size of improvement you want to find",
        yaxis_title="Visitors needed per version",
        xaxis_tickformat=".0%",
        yaxis_type="log",
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=GREY)
    fig.update_yaxes(gridcolor="#EEF1F4")
    st.plotly_chart(fig, width="stretch")

    half_lift = max(0.01, lift / 2)
    n_half = sample_size(
        baseline,
        treatment_rate(baseline, half_lift, "relative"),
        alpha=adjusted_alpha,
        power=power,
        method=method,
    )
    st.caption(
        f"The scale on the left doubles at every step. Halving your target from "
        f"{lift:.0%} to {half_lift:.0%} takes the visitors needed from "
        f"{n_needed:,} to {n_half:,}, roughly four times as many. Finding "
        "smaller effects is not slightly harder, it is dramatically harder."
    )


# --------------------------------------------------------------------------
# Reverse: what can I find?
# --------------------------------------------------------------------------

with tab_reverse:
    st.markdown(
        '<p class="lede">The question people actually ask. You have a fixed '
        "amount of traffic and a deadline. What is worth testing, and what is "
        "not worth starting?</p>",
        unsafe_allow_html=True,
    )

    st.write("")

    weeks_available = st.slider("Weeks you can give this test", 1, 12, 3)

    available_total = daily_visitors * allocation * weeks_available * 7
    n_available = int(available_total // n_variants)

    st.write("")

    try:
        found = detectable_effect(
            baseline,
            n_available,
            alpha=adjusted_alpha,
            power=power,
            method=method,
        )
        realised = achieved_power(
            baseline, found["treatment_rate"], n_available, alpha=adjusted_alpha, method=method
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("Visitors per version in that time", f"{n_available:,}")
        r2.metric("Smallest improvement you could find", f"{found['relative_mde']:.1%}")
        r3.metric(
            "Which means going from",
            f"{baseline:.2%}",
            delta=f"to {found['treatment_rate']:.2%}",
            delta_color="off",
        )

        st.markdown(
            f"""
            <div class="scenario">
            In {weeks_available} week{"s" if weeks_available > 1 else ""} you can
            reliably spot an improvement of <b>{found["relative_mde"]:.0%}</b> or
            better. Anything smaller than that will probably slip past you, and
            the test will end with no clear answer rather than a useful one.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        warnings = normal_approximation_warnings(
            baseline, found["treatment_rate"], n_available
        )
        if warnings:
            st.warning(
                "**These numbers are stretched thin.** "
                + " ".join(warnings)
                + " You do not have enough conversions for this estimate to mean much."
            )

        if found["relative_mde"] > 0.25:
            st.error(
                f"**A {found['relative_mde']:.0%} improvement is a lot to hope for.** "
                "Most product changes move conversion by a few percent at best. "
                "If this is all you can detect, the honest answer is that this "
                "test will not settle the question. Consider testing something "
                "bolder, or deciding without a test."
            )
        elif found["relative_mde"] < 0.05:
            st.success(
                f"**You are in good shape.** You have enough traffic to pick up "
                f"a {found['relative_mde']:.1%} improvement, which is a realistic "
                "size for a product change."
            )

        st.caption(
            f"At this sample size the test would catch a real improvement of that "
            f"size {realised:.0%} of the time, which is the target you set."
        )

    except ValueError as exc:
        st.error(f"**Not enough traffic.** {exc}")

st.divider()

st.markdown(
    """
    <div class="scenario" style="border-left-color:#E04E4E;">
    <b>One more thing.</b> Whatever number this page gives you, the plan only
    works if you let the test finish. Stopping the moment the dashboard looks
    good undoes the entire calculation. That is what the other page is about.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

with st.expander("What the words on this page mean"):
    st.markdown(
        """
**Smallest improvement worth finding.** Before starting, you decide how big a
change would actually matter to you. Set it small and the test needs enormous
traffic. Set it too large and you will miss real improvements. This is the most
important number on the page and the one most often picked carelessly.

**Improvement, measured how.** A 10% improvement on a 5% conversion rate means
5% to 5.5%. It does not mean 5% to 15%. Mixing these up changes the answer by
more than twenty times, and it is the single most common mistake with tools
like this one.

**How sure you want to be.** Your tolerance for shipping something that does
nothing. At 95% sure, roughly 1 test in 20 with no real effect will still look
like a winner.

**How often you want to catch a real improvement.** The other kind of mistake.
At 80%, a real improvement of the size you specified will still be missed about
1 time in 5. Raising this means more traffic and more days.

**Whole weeks.** Shopping behaviour on a Sunday is not the same as a Tuesday.
A test running 10 days counts two Mondays and one Saturday, which tilts the
result. Running in whole weeks gives every day equal weight.
"""
    )

with st.expander("Where these numbers come from"):
    n_alt = sample_size(
        baseline,
        target,
        alpha=adjusted_alpha,
        power=power,
        method="arcsine" if method == "pooled" else "pooled",
    )
    st.markdown(
        f"""
There are two accepted ways to work out a sample size for comparing two
conversion rates. They rest on slightly different approximations.

For the test you have set up:

- **Standard method:** {sample_size(baseline, target, alpha=adjusted_alpha, power=power, method="pooled"):,} visitors per version.
  This is what most commercial testing tools report.
- **Alternative method:** {sample_size(baseline, target, alpha=adjusted_alpha, power=power, method="arcsine"):,} visitors per version.
  This is what the widely used R package `pwr` reports.

For realistic tests the two agree to within a fraction of a percent. They only
pull apart when you are looking for very large improvements, several times the
starting rate, which is rare in practice.

Both are checked against their published formulas in the project's test suite,
along with a separate simulation that generates fake visitors and confirms the
recommended sample size really does catch a real improvement as often as
promised.
"""
    )
