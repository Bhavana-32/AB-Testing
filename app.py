"""
Entry point. Defines navigation and the styling shared by every page.

Pages live in `views/` rather than `pages/` on purpose: a folder literally
named `pages/` triggers Streamlit's automatic navigation, which would fight
with the explicit st.navigation setup below.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="A/B testing, honestly",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.4rem; max-width: 1180px; }
      h1 { font-size: 2.5rem !important; line-height: 1.15 !important;
           letter-spacing: -0.02em; margin-bottom: 0.2rem !important; }
      h2 { font-size: 1.45rem !important; letter-spacing: -0.01em;
           margin-top: 0.4rem !important; }
      .kicker { color: #69737F; font-size: 0.82rem; font-weight: 600;
                letter-spacing: 0.09em; text-transform: uppercase; }
      .lede { font-size: 1.12rem; line-height: 1.65; color: #2B3038;
              max-width: 62ch; }
      .scenario { background: #F7F8FA; border-left: 3px solid #3B6FD4;
                  padding: 1.1rem 1.3rem; border-radius: 4px;
                  font-size: 1.02rem; line-height: 1.65; color: #2B3038; }
      .punch { font-size: 1.3rem; line-height: 1.55; font-weight: 600;
               color: #16181D; max-width: 60ch; }
      div[data-testid="stMetricValue"] { font-size: 2.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

navigation = st.navigation(
    [
        st.Page(
            "views/peeking.py",
            title="Checking results early",
            icon=":material/visibility:",
            default=True,
        ),
        st.Page(
            "views/calculator.py",
            title="Plan a test",
            icon=":material/calculate:",
        ),
    ]
)

navigation.run()
