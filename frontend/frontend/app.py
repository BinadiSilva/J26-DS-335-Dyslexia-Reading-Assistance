from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Lumi-IT Reading", page_icon="⭐", layout="wide")

from frontend import data, kid_html
from frontend import logic
from frontend.theme import GLOBAL_CSS, LANDING_CSS
from frontend.view_parent import render_parent_view
from frontend.view_teacher import render_teacher_view

st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)

if "role" not in st.session_state:
    st.session_state.role = None

if st.session_state.role is None:
    st.markdown(f"<style>{LANDING_CSS}</style>", unsafe_allow_html=True)
    components.html(kid_html.welcome_page(), height=520)
    st.markdown("<h3 style='text-align:center;margin:.2rem 0 .6rem'>Who is using Lumi-IT today?</h3>", unsafe_allow_html=True)
    children = data.get_children()
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='who-card kid'><h4>🌟 Child / Parent</h4><span>See one child's reading profile on a single page.</span></div>",
                    unsafe_allow_html=True)
        if children:
            labels = {r["child_id"]: f'{r["name"]}  ({logic.display_id(r["child_id"])})' for r in children}
            cid = st.selectbox("Log in as", list(labels), format_func=lambda k: labels.get(k, k), key="login_child")
            if st.button("Log in as child / parent", use_container_width=True, type="primary"):
                st.session_state.role = "parent"
                st.session_state.child_id = cid
                st.rerun()
        else:
            st.warning("No sessions found. Run run_demo_payloads.py first.")
    with right:
        st.markdown("<div class='who-card tch'><h4>📋 Teacher / Specialist</h4><span>All students, their sessions and each session's cognitive profile.</span></div>",
                    unsafe_allow_html=True)
        if st.button("Log in as teacher / specialist", use_container_width=True):
            st.session_state.role = "teacher"
            st.rerun()
    st.caption("Demo role selector, not real authentication. Prototype built on synthetic demonstration data.")
else:
    top_l, _, top_r = st.columns([1, 4, 1])
    with top_l:
        if st.button("← Back", key="back"):
            st.session_state.role = None           # back to the main login page
            for k in ("child_id", "p_seen_listen", "p_plan_open"):
                st.session_state.pop(k, None)
            st.rerun()
    with top_r:
        if st.button("Log out", key="logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    if st.session_state.role == "parent":
        render_parent_view()
    else:
        render_teacher_view()
