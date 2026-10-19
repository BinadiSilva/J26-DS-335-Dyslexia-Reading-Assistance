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
/* readable text on light backgrounds, even if the browser/Streamlit theme is dark */
.stApp {color:#2b2350; color-scheme:light;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {color:#2b2350;}
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stWidgetLabel"] label, .stApp label p {color:#2b2350 !important; font-weight:700;}
.stApp [data-testid="stRadio"] label p, .stApp [data-testid="stRadio"] label div {color:#2b2350;}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p {color:#4d4577 !important;}
.stApp [data-testid="stAlert"] p {color:#1d3557 !important;}
.stApp .stTabs [data-baseweb="tab"] p {color:#3d4f6b;}
.stApp .stTabs [data-baseweb="tab"][aria-selected="true"] p {color:#d9363e;}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div {background:#ffffff; border:2px solid #d9d3f7; border-radius:14px;}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] div, .stApp [data-testid="stSelectbox"] [data-baseweb="select"] span {color:#2b2350 !important;}
.stApp [data-testid="stSelectbox"] svg {fill:#2b2350;}
[data-baseweb="popover"], [data-baseweb="popover"] *, [data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] *, ul[role="listbox"], ul[role="listbox"] * {background:#ffffff !important; background-color:#ffffff !important; color:#2b2350 !important;}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li:hover *, [data-baseweb="popover"] [aria-selected="true"], [data-baseweb="popover"] [aria-selected="true"] *, ul[role="listbox"] li:hover, ul[role="listbox"] li:hover * {background:#ece8ff !important; background-color:#ece8ff !important;}
[data-baseweb="popover"] {border-radius:14px; box-shadow:0 10px 26px rgba(70,60,140,.18);}
.stApp textarea {color:#2b2350 !important; background:#fff !important;}
"""

LANDING_CSS = """
.stApp {background:linear-gradient(180deg,#d6f0ff 0%,#fff3d6 55%,#ffe4ef 100%);}
.stApp h3 {color:#3b2a63;}
.block-container {padding-top:.8rem !important;}
.who-card {background:#fff; border-radius:22px; padding:1rem 1.2rem; box-shadow:0 10px 26px rgba(70,60,140,.12); margin-bottom:.6rem; border:3px solid #fff;}
.who-card h4 {margin:0 0 .2rem; color:#3b2a63;} .who-card span {color:#5b5380;}
.who-card.kid {border-color:#cdeee0;} .who-card.tch {border-color:#e1d9ff;}
"""

PARENT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;600;800&display=swap');
/* dyslexia-friendly reading: Lexend, wider letter/word/line spacing, left-aligned */
.stApp, .stApp button, .stApp label, .stApp p, .stApp li, .stApp [data-baseweb="select"] *, .stApp [data-testid="stCaptionContainer"] *, .gu-card, .gu-card *, .idrow, .idrow *, .review {font-family:'Lexend','Lexend Deca','Verdana','Trebuchet MS',system-ui,sans-serif;}
.gu-card, .idrow, .review, .stApp [data-testid="stCaptionContainer"] {letter-spacing:.03em; word-spacing:.1em; text-align:left;}
.gu-card p, .gu-card li, .gu-note, .main-skill span {line-height:1.8;}
.gu-card li {margin:.4rem 0;}
.gu-card h4 {line-height:1.5; margin-bottom:.6rem;}
.stApp {background:linear-gradient(180deg,#e9f7ff 0%,#fff6e0 100%);}
.gu-card {color:#3b2a63; background:#fff; border-radius:24px; padding:1.1rem 1.3rem; box-shadow:0 10px 26px rgba(70,60,140,.10); margin-bottom:1rem;}
.gu-card h4 {margin:0 0 .4rem; color:#3b2a63;}
.gu-hero {background:linear-gradient(135deg,#dff6ff,#f1e9ff); border-radius:24px; padding:1rem 1.3rem; margin-bottom:1rem;}
.gu-hero h2 {margin:0; color:#3b2a63;}
.gu-note {color:#6a608f; font-size:.85rem; margin-top:.35rem;}
.gu-card ul {margin:.3rem 0 .2rem 1.1rem; padding:0;} .gu-card li {margin:.25rem 0;} .gu-card p {margin:.35rem 0;}
.idrow {display:flex; flex-wrap:wrap; gap:.4rem 1.6rem; justify-content:space-between; align-items:center; background:#fff; border-radius:18px; padding:.7rem 1.1rem; margin:.2rem 0 1rem; box-shadow:0 6px 18px rgba(70,60,140,.08); color:#3b2a63;}
.idrow span {color:#6a608f; font-size:.9rem;}
.review {background:#fff3f2; border:1px solid #f3c4c0; color:#7a1d17; border-radius:14px; padding:.7rem 1rem; margin-bottom:1rem; font-weight:700;}
.pgrid {display:grid; grid-template-columns:1fr 1fr; gap:0 1rem;}
.big-row {display:flex; align-items:center; gap:.9rem; flex-wrap:wrap; margin:.2rem 0 .6rem;}
.lvl {font-size:1.5rem; font-weight:900; border-radius:999px; padding:.2rem 1rem;}
.lvl.g,.support.g {background:#e3f6ec; color:#17663a;} .lvl.a,.support.a {background:#fff1d6; color:#8a5800;} .lvl.r,.support.r {background:#fde4e4; color:#a11616;}
.score {font-weight:800; color:#3b2a63;} .score small {font-weight:600; color:#6a608f;}
.steps {display:flex; gap:.4rem; margin:.2rem 0 .5rem;} .step {flex:1; text-align:center; padding:.35rem 0; border-radius:12px; background:#ece8ff; color:#8a82b5; font-weight:800; font-size:.85rem;}
.step.g {background:#bfe9cf; color:#17663a;} .step.a {background:#ffe2a0; color:#8a5800;} .step.r {background:#f8c3c3; color:#a11616;}
.gauge {position:relative; display:flex; height:16px; border-radius:999px; overflow:hidden; margin-top:.5rem;}
.gauge .b1 {width:26%; background:#bfe9cf;} .gauge .b2 {width:24%; background:#ffe2a0;} .gauge .b3 {width:50%; background:#f8c3c3;}
.gauge i {position:absolute; top:-3px; width:6px; height:22px; margin-left:-3px; border-radius:4px; background:#3b2a63;}
.gl {display:flex; justify-content:space-between; font-size:.72rem; color:#6a608f; margin:.15rem 0 .4rem;}
.cert {background:#f4f1ff; border-radius:12px; padding:.5rem .8rem; margin-top:.5rem; font-size:.92rem; color:#3b2a63;}
.support {display:inline-block; font-size:1.45rem; font-weight:900; border-radius:16px; padding:.6rem 1.2rem; margin:.3rem 0 .4rem;}
.main-skill {background:#f4f1ff; border-radius:14px; padding:.7rem .9rem; font-size:1.1rem; color:#3b2a63;}
.main-skill span {display:block; font-size:.9rem; color:#6a608f; margin-top:.15rem;}
.chips2 {display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.3rem;}
.chip-w {background:#fde4e4; color:#a11616; border-radius:999px; padding:.25rem .7rem; font-weight:800; font-size:.88rem;}
.sk {margin:.6rem 0;} .skt {display:flex; justify-content:space-between; font-weight:800; color:#3b2a63;}
.skb {height:14px; background:#ece8ff; border-radius:999px; margin-top:.25rem; overflow:hidden;}
.skf {height:100%; border-radius:999px;} .skf.g {background:#2ca56a;} .skf.a {background:#e0a21f;} .skf.r {background:#e0645f;}
.expr {display:flex; justify-content:space-between; align-items:center; margin-top:.8rem; font-weight:800; color:#3b2a63;} .expr b {font-size:1.35rem; color:#f2b01e; letter-spacing:.1em;}
@media (max-width:820px){.pgrid{grid-template-columns:1fr;}}
"""

TEACHER_CSS = """
.stApp {background:#f3f8f7; color:#1f3a36;}
.stApp [data-testid="stMarkdownContainer"], .stApp .stMarkdown, .stApp [data-testid="stMarkdown"] {margin-bottom:0 !important;}
.stApp h4 {color:#16413c; font-size:1.08rem; line-height:1.35; margin:0; padding:.5rem 0 .2rem;}
.stApp [data-testid="stVerticalBlock"] {gap:.9rem;}
.block-container {padding-top:.6rem !important; padding-bottom:1.5rem;}
.stApp [data-testid="stWidgetLabel"] p {margin-bottom:.1rem;}
.t-title {font-size:1.7rem; font-weight:900; color:#16413c; margin:0;}
.t-sub {color:#5d7a76; margin:.2rem 0 .2rem;}
.t-card {color:#1f3a36; background:#fff; border:1px solid #dcebe8; border-radius:14px; padding:.9rem 1.1rem; margin-bottom:0;}
.t-card h4 {margin:0 0 .6rem; color:#16413c; font-size:1.02rem;}
.stat-strip {display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin-bottom:0;}
.stat {background:#fff; border:1px solid #dcebe8; border-radius:14px; padding:.6rem .9rem;}
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
.banner {border-radius:12px; padding:.6rem .9rem; margin:0; border:1px solid;}
.banner.red {background:#fff3f2; border-color:#f3c4c0; color:#7a1d17;} .banner.amber {background:#fff8e8; border-color:#f1dba5; color:#7a5200;}
.banner b {display:block; margin-bottom:.2rem;} .banner ul {margin:.2rem 0 0 1.1rem; padding:0;}
.kpis {display:grid; grid-template-columns:repeat(5,1fr); gap:.8rem; margin:0;}
.kpi {background:#fff; border:1px solid #dcebe8; border-radius:14px; padding:.6rem .8rem;}
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
.c4 {border:2px dashed #9fb6b2; background:#fbfdfc; border-radius:16px; padding:1.1rem 1.3rem; margin:.2rem 0;}
.c4-head {display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; border-bottom:1px solid #dcebe8; padding-bottom:.7rem; margin-bottom:.5rem;}
.c4-ttl {font-size:1.35rem; font-weight:900; color:#16413c;} .c4 h5 {margin:.8rem 0 .15rem; color:#16413c;} .c4 p {margin:.1rem 0; color:#1f3a36;}
.grid2 {display:grid; grid-template-columns:1fr 1fr; gap:.9rem;}
.act-li {padding:.45rem .7rem; background:#f0f8f6; border-radius:10px; margin:.35rem 0; color:#1f3a36;}
@media (max-width:900px){.kpis{grid-template-columns:repeat(2,1fr);} .stat-strip{grid-template-columns:repeat(2,1fr);} .grid2{grid-template-columns:1fr;}}
"""
