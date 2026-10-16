"""CSS for the Streamlit shell. Always injected with unsafe_allow_html=True."""
from __future__ import annotations

GLOBAL_CSS = """
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu {display:none !important;}
header[data-testid="stHeader"] {background:transparent;height:0;}
.block-container {padding-top:1.2rem !important; padding-bottom:3rem; max-width:1180px;}
html, body, [class*="css"], .stApp {font-family:'Nunito','Trebuchet MS','Segoe UI',system-ui,sans-serif;}
div.stButton > button {border-radius:999px; font-weight:800; padding:.6rem 1.4rem; border:2px solid #d9d3f7; color:#3b2a63; background:#fff;}
div.stButton > button:hover {border-color:#6b5bd6; color:#6b5bd6;}
div.stButton > button[kind="primary"] {background:linear-gradient(90deg,#6b5bd6,#4ecdc4); color:#fff; border:none; font-size:1.15rem; padding:.8rem 1.8rem; box-shadow:0 8px 18px rgba(107,91,214,.28);}
div.stButton > button[kind="primary"]:hover {color:#fff; transform:translateY(-2px);}
.stTabs [data-baseweb="tab"] {font-weight:800; font-size:1.02rem;}
"""

PARENT_CSS = """
.stApp {background:linear-gradient(180deg,#e9f7ff 0%,#fff6e0 100%);}
.gu-card {background:#fff; border-radius:24px; padding:1.1rem 1.3rem; box-shadow:0 10px 26px rgba(70,60,140,.10); margin-bottom:1rem;}
.gu-card h4 {margin:0 0 .4rem; color:#3b2a63;}
.gu-hero {background:linear-gradient(135deg,#dff6ff,#f1e9ff); border-radius:24px; padding:1rem 1.3rem; margin-bottom:1rem;}
.gu-hero h2 {margin:0; color:#3b2a63;}
.gu-note {color:#6a608f; font-size:.85rem;}
"""

TEACHER_CSS = """
.stApp {background:#f3f8f7;}
.t-title {font-size:1.7rem; font-weight:900; color:#16413c; margin:0;}
.t-sub {color:#5d7a76; margin:.1rem 0 1rem;}
.t-card {background:#fff; border:1px solid #dcebe8; border-radius:14px; padding:1rem 1.1rem; margin-bottom:.9rem;}
.t-card h4 {margin:0 0 .6rem; color:#16413c; font-size:1.02rem;}
.stat-strip {display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin-bottom:1rem;}
.stat {background:#fff; border:1px solid #dcebe8; border-radius:14px; padding:.8rem 1rem;}
.stat .l {font-size:.78rem; color:#5d7a76; font-weight:700;}
.stat .v {font-size:1.7rem; font-weight:900; color:#16413c;}
.stat .s {font-size:.75rem; color:#7d9692;}
.pill {display:inline-block; font-size:.74rem; font-weight:800; border-radius:999px; padding:.22rem .65rem; white-space:nowrap;}
.pill-green {background:#e3f6ec; color:#17663a;} .pill-amber {background:#fff1d6; color:#8a5800;}
.pill-red {background:#fde4e4; color:#a11616;} .pill-gray {background:#eceff1; color:#4b5b63;}
.pill-teal {background:#dff3f0; color:#15665d;} .pill-purple {background:#ece8ff; color:#4a3fb0;}
table.ct {width:100%; border-collapse:collapse; font-size:.92rem;}
table.ct th {text-align:left; font-size:.74rem; letter-spacing:.04em; text-transform:uppercase; color:#5d7a76; padding:.5rem .6rem; border-bottom:2px solid #dcebe8;}
table.ct td {padding:.6rem; border-bottom:1px solid #edf4f2; color:#1f3a36;}
table.ct tr.sel td {background:#eaf7f4;}
.child-head {display:flex; align-items:center; gap:1rem; flex-wrap:wrap;}
.avatar {width:56px; height:56px; border-radius:50%; background:linear-gradient(135deg,#2a9d8f,#79d3c2); color:#fff; font-weight:900; font-size:1.5rem; display:flex; align-items:center; justify-content:center;}
.child-head .nm {font-size:1.4rem; font-weight:900; color:#16413c;} .child-head .meta {color:#5d7a76; font-size:.85rem;}
.banner {border-radius:12px; padding:.8rem 1rem; margin:.6rem 0 1rem; border:1px solid;}
.banner.red {background:#fff3f2; border-color:#f3c4c0; color:#7a1d17;} .banner.amber {background:#fff8e8; border-color:#f1dba5; color:#7a5200;}
.banner b {display:block; margin-bottom:.2rem;} .banner ul {margin:.2rem 0 0 1.1rem; padding:0;}
.kpis {display:grid; grid-template-columns:repeat(5,1fr); gap:.8rem; margin:.4rem 0 1rem;}
.kpi {background:#fff; border:1px solid #dcebe8; border-radius:14px; padding:.8rem .9rem;}
.kpi .l {font-size:.74rem; color:#5d7a76; font-weight:800; text-transform:uppercase; letter-spacing:.03em;}
.kpi .v {font-size:1.5rem; font-weight:900; color:#16413c; margin:.15rem 0;} .kpi .s {font-size:.78rem; color:#5d7a76;}
.bar-row {margin:.65rem 0;} .bar-row .top {display:flex; justify-content:space-between; font-weight:800; color:#1f3a36; font-size:.92rem;}
.bar {position:relative; height:14px; background:#e9f0ef; border-radius:999px; margin-top:.3rem;}
.bar .fill {position:absolute; left:0; top:0; bottom:0; border-radius:999px;}
.bar .mk {position:absolute; top:-4px; bottom:-4px; width:2px; background:#5d7a76;}
.bar .mk span {position:absolute; top:-15px; left:-12px; font-size:.62rem; color:#5d7a76; font-weight:800;}
.chips {display:flex; flex-wrap:wrap; gap:.4rem; margin:.3rem 0;}
.sess-strip {display:flex; gap:.5rem; flex-wrap:wrap;} .sess {text-align:center; font-size:.7rem; color:#5d7a76;}
.sess .bx {display:block; border-radius:8px; padding:.35rem .5rem; font-weight:800; font-size:.74rem; margin-bottom:.15rem;}
.note {color:#5d7a76; font-size:.8rem;}
.grid2 {display:grid; grid-template-columns:1fr 1fr; gap:.9rem;}
.act-li {padding:.45rem .7rem; background:#f0f8f6; border-radius:10px; margin:.35rem 0; color:#1f3a36;}
@media (max-width:900px){.kpis{grid-template-columns:repeat(2,1fr);} .stat-strip{grid-template-columns:repeat(2,1fr);} .grid2{grid-template-columns:1fr;}}
"""
