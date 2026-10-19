"""HTML/SVG fragments for the teacher view (pure Python, no Streamlit)."""
from __future__ import annotations

import base64
import html

from . import logic

esc = html.escape
LEVEL_KIND = {"Low": "green", "Moderate": "amber", "High": "red"}


def svg_img(svg: str) -> str:
    """Embed an SVG as a data-URI image: renders identically everywhere (including Markdown)."""
    svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" font-family="Nunito,Segoe UI,Arial,sans-serif" ', 1)
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img alt="chart" style="width:100%;height:auto;display:block" src="data:image/svg+xml;base64,{b64}"/>'


def compact(s: str) -> str:
    """Single line, so Markdown never mistakes indented HTML for a code block."""
    return " ".join(s.split())


def humanize(name: str) -> str:
    return str(name).replace("_", " ").capitalize()


def pill(text: str, kind: str = "gray") -> str:
    return f'<span class="pill pill-{kind}">{esc(str(text))}</span>'


def level_pill(level: str) -> str:
    return pill(level, LEVEL_KIND.get(str(level).title(), "gray"))


def intensity_pill(v: str) -> str:
    """Support intensity as one of: Low / Medium / High support required."""
    label = logic.support_label(v)
    return pill(label, logic.SUPPORT_KIND[label])


def trend_pill(trend: str) -> str:
    return {"improving": pill("↗ Improving", "green"), "worsening": pill("↘ Worsening", "red")}.get(
        trend, pill("→ Stable", "gray"))


def review_pill(flag: bool) -> str:
    return pill("Review needed", "red") if flag else pill("Monitor", "gray")


def conf_text(conf: float, low: bool) -> str:
    return f"{float(conf) * 100:.0f}% " + (pill("Low confidence", "amber") if low else pill("OK", "green"))


def stat_strip(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(f'<div class="stat"><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>' for l, v, s in items)
    return compact(f'<div class="stat-strip">{cells}</div>')


def class_table(rows: list[dict], selected: str | None = None) -> str:
    body = []
    for r in rows:
        lt = r["latest"]
        body.append(
            f'<tr class="{"sel" if r["child_id"] == selected else ""}"><td><b>{esc(r["name"])}</b><div class="note">{esc(logic.display_id(r["child_id"]))}</div></td>'
            f'<td>{esc(str(r["age"]))} / {esc(str(r["grade"]))}</td><td>{r["n_sessions"]}</td><td>{esc(str(lt.get("timestamp", "")))}</td>'
            f'<td>{level_pill(lt.get("difficulty_level", ""))}</td>'
            f'<td>{intensity_pill(lt.get("intervention_intensity", ""))}</td>'
            f'<td>{trend_pill(r["trend"])}</td><td>{review_pill(r["review"])}</td></tr>')
    head = "".join(f"<th>{h}</th>" for h in ["Student", "Age / Grade", "Sessions", "Latest session", "Difficulty (latest)", "Support (latest)", "Trend", "Review"])
    return compact(f'<div class="t-card"><table class="ct"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def session_table(sessions: list[dict], selected: str | None = None) -> str:
    body = []
    for i, x in enumerate(sessions, 1):
        flag = bool(x.get("intervention", {}).get("requires_professional_review"))
        body.append(
            f'<tr class="{"sel" if x.get("session_id") == selected else ""}"><td><b>Session {i:02d}</b><div class="note">{esc(logic.display_id(x.get("session_id", "")))}</div></td>'
            f'<td>{esc(str(x.get("timestamp", "")))}</td><td>{float(x.get("difficulty_score", 0)):.2f} {level_pill(x.get("difficulty_level", ""))}</td>'
            f'<td>{esc(str(x.get("weak_skill", "")).title())}</td><td>{intensity_pill(x.get("intervention_intensity", ""))}</td><td>{review_pill(flag)}</td></tr>')
    head = "".join(f"<th>{h}</th>" for h in ["Session", "Date", "Difficulty", "Weak skill", "Support intensity", "Review"])
    return compact(f'<div class="t-card"><h4>Assessment sessions</h4><table class="ct"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def component4_box(name: str, session_line: str) -> str:
    return compact(
        '<div class="c4"><div class="c4-head"><div><div class="c4-ttl">📈 Learning Progress Prediction</div>'
        '<div class="note">Component 04 · illustrative integration placeholder</div></div>'
        f'{pill("Pending integration", "gray")}</div>'
        '<h5>Purpose</h5><p>Predict the learner\'s future reading progress from the assessment history and the other inputs defined by Component 04.</p>'
        f'<h5>Input from Component 02</h5><p>The assessed profile for <b>{esc(name)}</b> ({esc(session_line)}) and the session history shown above.</p>'
        '<h5>Expected output</h5><p>Learning-progress prediction results from Component 04.</p>'
        '<div class="note">No Component 04 results are shown here. Progress indicators, prediction horizon and charts will follow Component 04\'s own design.</div></div>')


def child_header(info: dict, summ: dict, review: bool) -> str:
    return compact(
        f'<div class="t-card"><div class="child-head"><div class="avatar">{esc(str(info["name"]).split()[-1][:1])}</div>'
        f'<div><div class="nm">{esc(info["name"])} <span class="meta">· {esc(logic.display_id(info["child_id"]))}</span></div>'
        f'<div class="meta">Age {esc(str(info["age"]))} · Grade {esc(str(info["grade"]))} · {summ["n_sessions"]} sessions</div></div>'
        f'<div style="margin-left:auto;display:flex;gap:.4rem;flex-wrap:wrap">{trend_pill(summ["trend"])}{review_pill(review)}</div></div></div>')


def banner(kind: str, title: str, items: list[str]) -> str:
    lis = "".join(f"<li>{esc(i)}</li>" for i in items)
    return compact(f'<div class="banner {kind}"><b>{esc(title)}</b><ul>{lis}</ul></div>')


def kpi_cards(s: dict) -> str:
    flags = s.get("low_confidence_flags", {})
    pros = s.get("prosody_engagement_index", {}).get("prosody_engagement_index")
    cards = [
        ("Difficulty", f'{float(s.get("difficulty_score", 0)):.2f}', level_pill(s.get("difficulty_level", ""))),
        ("Weak skill", esc(str(s.get("weak_skill", "")).title()), conf_text(s.get("weak_skill_confidence", 0), flags.get("weak_skill", False))),
        ("Support intensity", esc(logic.support_label(s.get("intervention_intensity", ""))), conf_text(s.get("intervention_confidence", 0), flags.get("intervention_intensity", False))),
        ("Reading expression", "n/a" if pros is None else f"{float(pros):.0f}/100", '<span class="note">Not clinically validated</span>'),
        ("Session", esc(logic.display_id(s.get("session_id", "")).split("-")[-1]), f'<span class="note">{esc(str(s.get("timestamp", "")))}</span>'),
    ]
    return compact('<div class="kpis">' + "".join(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{sub}</div></div>' for l, v, sub in cards) + "</div>")


def skill_bars(scores: dict) -> str:
    out = []
    for k in ["phonological", "decoding", "fluency", "pronunciation"]:
        v = max(0.0, min(100.0, float(scores.get(k, 0))))
        col = "#2ca56a" if v >= 75 else ("#e0a21f" if v >= 60 else "#d9534f")
        out.append(
            f'<div class="bar-row"><div class="top"><span>{k.title()}</span><span>{v:.0f}</span></div>'
            f'<div class="bar"><div class="fill" style="width:{v}%;background:{col}"></div>'
            f'<div class="mk" style="left:60%"><span>60</span></div><div class="mk" style="left:75%"><span>75</span></div></div></div>')
    return compact('<div style="padding-top:.8rem">' + "".join(out) + '</div><div class="note">Below 60 = weakness, 75 or above = strength (rule-based).</div>')


def chips(items: list[str], kind: str = "gray", empty: str = "None") -> str:
    if not items:
        return f'<span class="note">{esc(empty)}</span>'
    return compact('<div class="chips">' + "".join(pill(str(i).title(), kind) for i in items) + "</div>")


def activities(items: list[str]) -> str:
    return compact("".join(f'<div class="act-li">{esc(i)}</div>' for i in items)) or '<span class="note">None</span>'


def component_bars(components: dict) -> str:
    out = []
    for k, v in components.items():
        v = max(0.0, min(100.0, float(v)))
        out.append(f'<div class="bar-row"><div class="top"><span>{esc(humanize(k))}</span><span>{v:.0f}</span></div><div class="bar"><div class="fill" style="width:{v}%;background:#2a9d8f"></div></div></div>')
    return compact("".join(out))


def session_strip(sessions: list[dict], key: str, kinds: dict | None = None) -> str:
    palette = {"phonological": "#ece8ff", "decoding": "#dff3f0", "fluency": "#fff1d6", "pronunciation": "#fde4e4",
               "Low": "#e3f6ec", "Moderate": "#fff1d6", "High": "#fde4e4"}
    cells = []
    for i, s in enumerate(sessions, 1):
        v = str(s.get(key, ""))
        shown = logic.support_label(v).replace(" required", "") if key == "intervention_intensity" else v.title()
        cells.append(f'<div class="sess"><span class="bx" style="background:{palette.get(v, "#eceff1")}">{esc(shown)}</span>S{i}</div>')
    return compact('<div class="sess-strip">' + "".join(cells) + "</div>")


def line_chart_svg(values: list[float], labels: list[str], *, ymin: float = 0, ymax: float = 1,
                   bands: list[tuple] | None = None, color: str = "#2a9d8f", fmt: str = "{:.2f}",
                   w: int = 660, h: int = 250) -> str:
    ml, mr, mt, mb = 44, 78, 18, 34
    pw, ph = w - ml - mr, h - mt - mb

    def X(i): return ml + (pw / 2 if len(values) == 1 else i * pw / (len(values) - 1))
    def Y(v): return mt + ph * (1 - (min(max(v, ymin), ymax) - ymin) / (ymax - ymin))

    g = []
    for lo, hi, c, lab in (bands or []):
        g.append(f'<rect x="{ml}" y="{Y(hi):.1f}" width="{pw}" height="{Y(lo) - Y(hi):.1f}" fill="{c}" opacity=".55"/>'
                 f'<text x="{ml + pw + 8}" y="{(Y(lo) + Y(hi)) / 2 + 4:.1f}" font-size="11" fill="#5d7a76" font-weight="700">{esc(lab)}</text>')
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        g.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#dbe7e4" stroke-width="1"/>'
                 f'<text x="{ml - 8}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#5d7a76">{fmt.format(v)}</text>')
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(values))
    if len(values) > 1:
        g.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
    for i, v in enumerate(values):
        g.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5" fill="#fff" stroke="{color}" stroke-width="3"><title>{esc(labels[i])}: {fmt.format(v)}</title></circle>'
                 f'<text x="{X(i):.1f}" y="{Y(v) - 11:.1f}" font-size="11" text-anchor="middle" font-weight="800" fill="#16413c">{fmt.format(v)}</text>'
                 f'<text x="{X(i):.1f}" y="{h - 10}" font-size="11" text-anchor="middle" fill="#5d7a76">{esc(labels[i])}</text>')
    return svg_img(compact(f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">{"".join(g)}</svg>'))


def signed_bars_svg(items: list[tuple[str, float]], w: int = 660) -> str:
    if not items:
        return '<span class="note">No explanation available.</span>'
    lw, rh = 200, 30
    h = rh * len(items) + 10
    cx = lw + (w - lw) / 2
    half = (w - lw) / 2 - 56
    mx = max(abs(float(v)) for _, v in items) or 1.0
    g = [f'<line x1="{cx}" x2="{cx}" y1="0" y2="{h}" stroke="#9fb6b2" stroke-width="1.5"/>']
    for i, (name, v) in enumerate(items):
        v = float(v)
        y = 6 + i * rh
        bw = abs(v) / mx * half
        x = cx if v >= 0 else cx - bw
        col = "#2a9d8f" if v >= 0 else "#e07a5f"
        tx = cx + bw + 6 if v >= 0 else cx - bw - 6
        g.append(f'<text x="{lw - 10}" y="{y + 15}" font-size="12.5" text-anchor="end" fill="#1f3a36">{esc(humanize(name))}</text>'
                 f'<rect x="{x:.1f}" y="{y}" width="{bw:.1f}" height="20" rx="5" fill="{col}"/>'
                 f'<text x="{tx:.1f}" y="{y + 15}" font-size="11.5" font-weight="800" text-anchor="{"start" if v >= 0 else "end"}" fill="#1f3a36">{v:+.3f}</text>')
    return svg_img(compact(f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">{"".join(g)}</svg>'))
