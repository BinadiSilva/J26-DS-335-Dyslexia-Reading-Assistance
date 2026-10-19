"""Kid-friendly storybook pages as self-contained HTML (rendered in an iframe by Streamlit).
Original mascot: 'Lumi', a friendly little star who loves books. Pure HTML/CSS/SVG, works offline."""
from __future__ import annotations

import html
import math
import random

from . import copy

def _star_points(cx=100, cy=96, outer=74, inner=38) -> str:
    pts = []
    for i in range(10):
        r = outer if i % 2 == 0 else inner
        a = math.radians(-90 + i * 36)
        pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
    return " ".join(pts)


def lumi_svg(size: int = 230, cheer: bool = True) -> str:
    star = _star_points()
    return f"""
<svg class="lumi" width="{size}" height="{int(size*1.05)}" viewBox="0 0 200 210" role="img" aria-label="Lumi the friendly star reading a book">
  <ellipse cx="100" cy="204" rx="52" ry="6" fill="rgba(60,50,120,.15)" class="shadow"/>
  <g class="bounce">
    <polygon points="{star}" fill="#ffd84d" stroke="#ffd84d" stroke-width="16" stroke-linejoin="round"/>
    <polygon points="{star}" fill="none" stroke="#f4b400" stroke-width="3" stroke-linejoin="round" opacity=".55" transform="translate(0,2)"/>
    <ellipse cx="68" cy="108" rx="9" ry="6" fill="#ff9db8" opacity=".75"/>
    <ellipse cx="132" cy="108" rx="9" ry="6" fill="#ff9db8" opacity=".75"/>
    <g class="eyes">
      <ellipse cx="85" cy="94" rx="6.5" ry="8.5" fill="#3b2a63"/><circle cx="87.5" cy="90.5" r="2.6" fill="#fff"/>
      <ellipse cx="115" cy="94" rx="6.5" ry="8.5" fill="#3b2a63"/><circle cx="117.5" cy="90.5" r="2.6" fill="#fff"/>
    </g>
    <path d="M86 112 Q100 128 114 112" stroke="#3b2a63" stroke-width="4.5" fill="#fff" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M72 128 Q60 142 58 156" stroke="#ffd84d" stroke-width="12" stroke-linecap="round" fill="none"/>
    <path d="M128 128 Q140 142 142 156" stroke="#ffd84d" stroke-width="12" stroke-linecap="round" fill="none"/>
    <path d="M52 150 L52 188 Q78 180 100 191 Q122 180 148 188 L148 150 Z" fill="#6b5bd6"/>
    <path d="M100 154 Q80 142 56 148 L56 184 Q80 177 100 188 Z" fill="#fff" stroke="#e3defa" stroke-width="2"/>
    <path d="M100 154 Q120 142 144 148 L144 184 Q120 177 100 188 Z" fill="#fff" stroke="#e3defa" stroke-width="2"/>
    <path d="M66 160 Q80 156 92 162 M66 169 Q80 165 92 171 M108 162 Q120 156 134 160 M108 171 Q120 165 134 169" stroke="#c9c2f0" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    <circle cx="58" cy="156" r="7.5" fill="#ffd84d"/><circle cx="142" cy="156" r="7.5" fill="#ffd84d"/>
  </g>
  <g class="sparkles" fill="#fff" stroke="#ffb703" stroke-width="1.5">
    <path class="sp s1" d="M20 40 l4 10 10 4 -10 4 -4 10 -4 -10 -10 -4 10 -4z"/>
    <path class="sp s2" d="M176 30 l3 8 8 3 -8 3 -3 8 -3 -8 -8 -3 8 -3z"/>
    <path class="sp s3" d="M180 120 l3 8 8 3 -8 3 -3 8 -3 -8 -8 -3 8 -3z"/>
  </g>
</svg>"""


def _confetti(n: int = 26) -> str:
    rnd = random.Random(7)
    colors = ["#ff6b6b", "#ffd84d", "#4ecdc4", "#6b5bd6", "#ff9db8", "#7bd88f"]
    bits = []
    for i in range(n):
        left = rnd.randint(2, 98)
        delay = rnd.random() * 1.4
        dur = 2.6 + rnd.random() * 1.6
        size = rnd.randint(7, 12)
        shape = "50%" if i % 3 == 0 else "2px"
        bits.append(
            f'<i class="cf" style="left:{left}%;width:{size}px;height:{size}px;border-radius:{shape};'
            f'background:{colors[i % len(colors)]};animation-delay:{delay:.2f}s;animation-duration:{dur:.2f}s"></i>')
    return "".join(bits)


CSS = """
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{font-family:'Nunito','Trebuchet MS','Segoe UI','Arial Rounded MT Bold',system-ui,sans-serif;color:#2b2350;overflow:hidden}
.sky{position:relative;height:100%;min-height:560px;overflow:hidden;
 background:linear-gradient(180deg,#bfe8ff 0%,#e2f6ff 42%,#fff2d1 100%)}
.sun{position:absolute;left:3%;top:5%;width:92px;height:92px;border-radius:50%;background:#ffd84d;box-shadow:0 0 0 12px rgba(255,216,77,.35),0 0 0 26px rgba(255,216,77,.18);animation:pulse 3.4s ease-in-out infinite}
.sun:before,.sun:after{content:"";position:absolute;top:36px;width:9px;height:12px;border-radius:50%;background:#3b2a63}
.sun:before{left:24px}.sun:after{right:24px}
.sun i{position:absolute;left:30px;top:54px;width:32px;height:16px;border-bottom:5px solid #3b2a63;border-radius:0 0 32px 32px}
.cloud{position:absolute;background:#fff;border-radius:60px;opacity:.95;animation:drift linear infinite}
.cloud:before,.cloud:after{content:"";position:absolute;background:#fff;border-radius:50%}
.c1{top:9%;width:120px;height:38px;animation-duration:38s;left:-160px}
.c1:before{width:52px;height:52px;left:18px;top:-24px}.c1:after{width:62px;height:62px;left:52px;top:-32px}
.c2{top:24%;width:90px;height:30px;animation-duration:52s;animation-delay:-20s;left:-160px}
.c2:before{width:40px;height:40px;left:14px;top:-18px}.c2:after{width:46px;height:46px;left:38px;top:-24px}
.hill{position:absolute;bottom:-70px;border-radius:50%}
.h1{left:-8%;width:62%;height:190px;background:#9ddf9a}.h2{right:-10%;width:66%;height:170px;background:#7fd38a}
.rainbow{position:absolute;right:5%;bottom:70px;width:170px;height:85px;overflow:hidden;opacity:.9}
.rainbow:before{content:"";position:absolute;left:0;top:0;width:170px;height:170px;border-radius:50%;
 border:12px solid #ff7b7b;box-shadow:inset 0 0 0 12px #ffc26b,inset 0 0 0 24px #fff08a,inset 0 0 0 36px #8fe39a,inset 0 0 0 48px #8ec9ff}
.blk{position:absolute;width:54px;height:54px;border-radius:14px;display:flex;align-items:center;justify-content:center;
 font-weight:900;font-size:30px;color:#fff;box-shadow:0 6px 0 rgba(0,0,0,.12);animation:bob 3s ease-in-out infinite}
.b1{left:5%;bottom:24%;background:#ff6b6b;transform:rotate(-8deg)}
.b2{left:13%;bottom:14%;background:#4ecdc4;animation-delay:.6s;transform:rotate(6deg)}
.b3{right:6%;bottom:17%;background:#6b5bd6;animation-delay:1.2s;transform:rotate(9deg)}
.card{position:relative;z-index:2;margin:34px auto 0;width:min(900px,92%);background:rgba(255,255,255,.95);border-radius:36px;
 box-shadow:0 18px 40px rgba(70,60,140,.18);padding:22px 28px 18px;display:flex;flex-wrap:wrap;gap:12px 26px;align-items:center;justify-content:center;
 animation:pop .6s cubic-bezier(.2,1.4,.4,1) both}
.mascot{flex:0 0 auto;width:230px;text-align:center}
.content{flex:1 1 340px;min-width:280px}
h1{margin:0 0 6px;font-size:34px;line-height:1.1;color:#3b2a63;font-weight:900}
h2{margin:0 0 6px;font-size:28px;color:#3b2a63;font-weight:900}
.sub{font-size:19px;line-height:1.35;color:#5a5088;margin:4px 0 12px;font-weight:700}
.tag{display:inline-block;background:#ffe9a8;color:#7a5200;font-weight:900;border-radius:999px;padding:6px 14px;font-size:15px;margin-bottom:8px}
.track{position:relative;margin:34px 6px 8px;height:20px;border-radius:999px;background:linear-gradient(90deg,#ff9a8b,#ffd36b 50%,#7bd88f)}
.track .stop{position:absolute;top:-4px;width:28px;height:28px;border-radius:50%;background:#fff;border:5px solid #fff;box-shadow:0 2px 0 rgba(0,0,0,.12)}
.labels{display:flex;justify-content:space-between;font-weight:800;font-size:14px;color:#6a608f;margin:4px 0 0}
.rider{position:absolute;top:-30px;left:0;width:48px;height:48px;margin-left:-24px;animation:ride 1.8s cubic-bezier(.3,1.2,.4,1) .5s forwards;left:var(--from,2%)}
.rider svg{width:48px;height:48px;animation:bob 1.6s ease-in-out infinite}
.badge{width:150px;height:150px;border-radius:50%;margin:0 auto 6px;display:flex;align-items:center;justify-content:center;font-size:72px;
 background:radial-gradient(circle at 35% 30%,#fff7c9,#ffd84d 60%,#ffb703);box-shadow:0 0 0 10px rgba(255,216,77,.3),0 12px 24px rgba(255,183,3,.4);animation:spin-in .9s cubic-bezier(.2,1.3,.4,1) both}
.big{font-size:32px;font-weight:900;color:#3b2a63;margin:8px 0 0}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}
.chip{background:#eef0ff;border-radius:18px;padding:10px 16px;font-weight:800;font-size:18px;color:#3b2a63;animation:pop .5s both}
.act{display:flex;gap:14px;align-items:center;background:#f4f1ff;border-radius:20px;padding:12px 16px;margin:10px 0;font-size:19px;font-weight:800;color:#3b2a63;animation:slide .6s both}
.act .n{flex:0 0 38px;height:38px;border-radius:50%;background:#6b5bd6;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:900}
.stars{display:flex;gap:6px;margin:10px 0}
.stars span{font-size:56px;line-height:1;animation:starpop .6s cubic-bezier(.2,1.6,.4,1) both;filter:drop-shadow(0 4px 0 rgba(0,0,0,.08))}
.stars .off{filter:grayscale(1) opacity(.25)}
.dots{width:100%;display:flex;justify-content:center;gap:10px;margin-top:2px}
.dots b{width:12px;height:12px;border-radius:50%;background:#d9d3f7}.dots b.on{width:34px;border-radius:12px;background:#6b5bd6}
.cf{position:absolute;top:-20px;z-index:1;animation:fall linear 2 both}
.lumi .bounce{animation:bounce 1.15s ease-in-out infinite alternate;transform-origin:100px 190px}
.lumi .shadow{animation:shadow 1.15s ease-in-out infinite alternate;transform-origin:100px 204px}
.lumi .eyes{animation:blink 4.2s infinite;transform-origin:100px 94px}
.lumi .sp{animation:twinkle 1.8s ease-in-out infinite;transform-origin:center;transform-box:fill-box}
.lumi .s2{animation-delay:.5s}.lumi .s3{animation-delay:1s}
@keyframes bounce{from{transform:translateY(0) rotate(-2deg)}to{transform:translateY(-12px) rotate(2deg)}}
@keyframes shadow{from{transform:scaleX(1);opacity:.9}to{transform:scaleX(.86);opacity:.6}}
@keyframes blink{0%,92%,100%{transform:scaleY(1)}95%{transform:scaleY(.1)}}
@keyframes twinkle{0%,100%{transform:scale(.6) rotate(0);opacity:.4}50%{transform:scale(1.15) rotate(20deg);opacity:1}}
@keyframes drift{from{transform:translateX(0)}to{transform:translateX(1300px)}}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes pop{from{transform:scale(.8);opacity:0}to{transform:scale(1);opacity:1}}
@keyframes slide{from{transform:translateX(-30px);opacity:0}to{transform:none;opacity:1}}
@keyframes starpop{from{transform:scale(0) rotate(-40deg);opacity:0}to{transform:scale(1);opacity:1}}
@keyframes spin-in{from{transform:scale(.2) rotate(-180deg);opacity:0}to{transform:none;opacity:1}}
@keyframes ride{to{left:var(--pos)}}
@keyframes fall{from{transform:translateY(0) rotate(0)}to{transform:translateY(640px) rotate(540deg)}}
@media (prefers-reduced-motion:reduce){*{animation:none!important}.rider{left:var(--pos)}}
"""


def _doc(inner: str, confetti: bool = False) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<style>{CSS}</style></head><body><div class="sky"><div class="sun"><i></i></div>'
            f'<div class="cloud c1"></div><div class="cloud c2"></div><div class="rainbow"></div>'
            f'<div class="hill h1"></div><div class="hill h2"></div>'
            f'<div class="blk b1">A</div><div class="blk b2">B</div><div class="blk b3">C</div>'
            f'{_confetti() if confetti else ""}{inner}</div></body></html>')


def profile_hero(name: str, session_line: str, level: str) -> str:
    """Short friendly header for the one-page profile (mascot + greeting + session)."""
    headline, _pos, kid_sentence, _ = copy.level_parts(level)
    inner = f"""
<div class="card" style="margin-top:22px"><div class="mascot" style="width:170px">{lumi_svg(160)}</div>
<div class="content"><span class="tag">{html.escape(session_line)}</span>
<h1>Hi, {html.escape(name)}!</h1>
<div class="sub">{html.escape(headline)} {html.escape(kid_sentence)}</div></div></div>"""
    return _doc(inner, confetti=True).replace("</style>", ".sky{min-height:0!important}</style>", 1)


def listening_page(name: str) -> str:
    bars = "".join(f'<i style="animation-delay:{i*0.09:.2f}s"></i>' for i in range(14))
    extra = """
.wave{display:flex;gap:8px;align-items:center;justify-content:center;height:90px;margin:14px 0}
.wave i{width:12px;border-radius:8px;background:linear-gradient(#6b5bd6,#4ecdc4);animation:wv 0.9s ease-in-out infinite;height:20px}
@keyframes wv{0%,100%{height:16px}50%{height:84px}}
"""
    inner = f"""
<div class="card"><div class="mascot">{lumi_svg()}</div>
<div class="content"><span class="tag">Lumi is listening</span>
<h2>Reading with {html.escape(name)}...</h2>
<div class="wave">{bars}</div>
<div class="sub">Lumi is working out the reading story. Just a moment!</div></div></div>"""
    return _doc(inner).replace("</style>", extra + "</style>")


def welcome_page(name: str | None = None) -> str:
    who = f", {html.escape(name)}" if name else ""
    inner = f"""
<div class="card"><div class="mascot">{lumi_svg()}</div>
<div class="content"><span class="tag">Welcome to Lumi-IT</span>
<h1>Hi{who}! Ready to read?</h1>
<div class="sub">Choose who is using Lumi-IT below. Lumi will show how reading is going.</div></div></div>"""
    return _doc(inner)
