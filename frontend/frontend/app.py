from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Lumi-It Reading", page_icon="⭐", layout="wide")

from frontend import kid_html
from frontend.theme import GLOBAL_CSS
from frontend.view_parent import render_parent_view
from frontend.view_teacher import render_teacher_view

st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)

if "role" not in st.session_state:
    st.session_state.role = None

if st.session_state.role is None:
    components.html(kid_html.welcome_page(), height=520)
    st.markdown("<h3 style='text-align:center;margin:.2rem 0 .6rem'>Who is using Lumi-It today?</h3>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("<div style='background:#fff;border-radius:22px;padding:1rem 1.2rem;box-shadow:0 10px 26px rgba(70,60,140,.10);margin-bottom:.6rem'>"
                    "<h4 style='margin:0'>🌟 Child / Parent</h4><span style='color:#6a608f'>A fun reading summary for the child and a clear explanation for grown-ups.</span></div>",
                    unsafe_allow_html=True)
        if st.button("Child / Parent", use_container_width=True, type="primary"):
            st.session_state.role = "parent"
            st.rerun()
    with right:
        st.markdown("<div style='background:#fff;border-radius:22px;padding:1rem 1.2rem;box-shadow:0 10px 26px rgba(70,60,140,.10);margin-bottom:.6rem'>"
                    "<h4 style='margin:0'>📋 Teacher / Specialist</h4><span style='color:#6a608f'>Class overview, detailed profiles, progress over time and explanations.</span></div>",
                    unsafe_allow_html=True)
        if st.button("Teacher / Specialist", use_container_width=True):
            st.session_state.role = "teacher"
            st.rerun()
    st.caption("Demo login, not secure. Prototype built on demonstration data.")
else:
    top_l, _, top_r = st.columns([1, 4, 1])
    with top_r:
        if st.button("Log out", key="logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    if st.session_state.role == "parent":
        render_parent_view()
    else:
        render_teacher_view()
