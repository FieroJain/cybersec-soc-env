# -*- coding: utf-8 -*-
"""
server/gradio_dashboard.py — Professional CyberSec-SOC monitoring dashboard.

Mounted via:
    import gradio as gr
    from .gradio_dashboard import demo
    gr.mount_gradio_app(app, demo, path="/web")

Features:
  - Live networkx/matplotlib network topology graph (dark theme, glowing nodes)
  - Control panel: Reset | Scan | Isolate | Patch | Firewall | Nothing
  - Live stats panel: attack stage, business impact, timestep, threats
  - SIEM-style scrolling alert feed
  - Episode score with colour-coded reward
"""

from __future__ import annotations

import threading
from typing import Optional

import gradio as gr
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must be before pyplot
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .soc_environment import SOCEnvironment, TASK_CONFIG
from ..models import SOCAction, SOCObservation

# ── Thread-safe global dashboard state ────────────────────────────────────────
_lock        = threading.Lock()
_env:  Optional[SOCEnvironment]  = None
_obs:  Optional[SOCObservation]  = None
_total_reward: float             = 0.0
_graph_pos:    Optional[dict]    = None   # spring-layout cache

# ── Attack stage label mapping ─────────────────────────────────────────────────
_STAGE = {
    1: ("🟡", "Initial Compromise"),
    2: ("🟠", "Credential Access"),
    3: ("🔴", "Lateral Movement"),
    4: ("💀", "Exfiltration"),
}

# ══════════════════════════════════════════════════════════════════════════════
#  NETWORK GRAPH RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _draw_network() -> plt.Figure:
    """Render the live network topology as a dark-themed matplotlib Figure."""
    global _env, _obs, _graph_pos

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0b0f19")
    ax.set_facecolor("#0b0f19")

    # ── Splash screen when no episode ─────────────────────────────────────────
    if _env is None or _obs is None:
        ax.text(
            0.5, 0.54,
            "🛡  CyberSec-SOC",
            ha="center", va="center",
            color="#00aaff", fontsize=20, fontweight="bold",
            fontfamily="monospace", transform=ax.transAxes,
        )
        ax.text(
            0.5, 0.44,
            "Press  ⟳ RESET  to start a new episode",
            ha="center", va="center",
            color="#3a6a8a", fontsize=11,
            fontfamily="monospace", transform=ax.transAxes,
        )
        ax.axis("off")
        return fig

    G = _env._graph
    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center",
                color="#a0b0c0", transform=ax.transAxes)
        ax.axis("off")
        return fig

    # ── Layout (cached per episode) ────────────────────────────────────────────
    if _graph_pos is None or len(_graph_pos) != len(G.nodes):
        k = 2.8 / max(1, len(G.nodes) ** 0.5)
        _graph_pos = nx.spring_layout(G, seed=42, k=k)
    pos = _graph_pos

    true_compromised = set(_env.state.true_compromised)

    # ── Per-node colours ───────────────────────────────────────────────────────
    node_colors  = []
    border_colors = []
    node_sizes   = []

    for nid in G.nodes:
        nd   = G.nodes[nid]
        size = 900 + int(nd["asset_value"] * 1600)
        node_sizes.append(size)

        isolated = nd["isolated"]
        compromised = nid in true_compromised
        scanned  = nd.get("scanned", False)
        alert_s  = float(_obs.node_statuses[nid]["alert_score"] if nid < len(_obs.node_statuses) else 0.5)

        if isolated:
            node_colors.append("#d06000")
            border_colors.append("#ff9933")
        elif compromised and scanned:      # detected active threat
            node_colors.append("#cc0022")
            border_colors.append("#ff3355")
        elif compromised:                  # hidden threat
            node_colors.append("#660011")
            border_colors.append("#990022")
        elif scanned:                      # confirmed clean
            node_colors.append("#006688")
            border_colors.append("#00ccff")
        elif alert_s > 0.5:               # high alert — suspicious
            node_colors.append("#4a3000")
            border_colors.append("#cc8800")
        else:                             # unknown
            node_colors.append("#152535")
            border_colors.append("#1e4060")

    # ── Draw edges ─────────────────────────────────────────────────────────────
    if len(G.edges) > 0:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color="#1a3a5a", width=1.3, alpha=0.55, arrows=False,
        )

    # ── Glow halos behind each node ────────────────────────────────────────────
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=border_colors,
        node_size=[s * 1.55 for s in node_sizes],
        alpha=0.18, edgecolors="none",
    )

    # ── Main nodes ─────────────────────────────────────────────────────────────
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
        edgecolors=border_colors,
        linewidths=1.8,
    )

    # ── Labels ─────────────────────────────────────────────────────────────────
    nx.draw_networkx_labels(
        G, pos,
        labels={nid: str(nid) for nid in G.nodes},
        ax=ax,
        font_color="#ffffff", font_size=8,
        font_weight="bold", font_family="monospace",
    )

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor="#660011", edgecolor="#990022", label="Hidden threat"),
        mpatches.Patch(facecolor="#cc0022", edgecolor="#ff3355", label="Detected threat"),
        mpatches.Patch(facecolor="#d06000", edgecolor="#ff9933", label="Isolated"),
        mpatches.Patch(facecolor="#4a3000", edgecolor="#cc8800", label="Suspicious"),
        mpatches.Patch(facecolor="#006688", edgecolor="#00ccff", label="Scanned / clean"),
        mpatches.Patch(facecolor="#152535", edgecolor="#1e4060", label="Unknown"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower left",
        facecolor="#0d1520", edgecolor="#1e3a5f",
        labelcolor="#90b0c8", fontsize=7.5, framealpha=0.92,
    )

    # ── Title ──────────────────────────────────────────────────────────────────
    stage_emoji, stage_name = _STAGE.get(_obs.attack_stage, ("❓", "Unknown"))
    ax.set_title(
        f"Topology: {_obs.topology_type.upper()}   |   "
        f"Nodes: {len(G.nodes)}   |   "
        f"Stage {_obs.attack_stage}: {stage_emoji} {stage_name}",
        color="#00aaff", fontsize=10, fontfamily="monospace",
        fontweight="bold", pad=12,
    )
    ax.axis("off")
    fig.tight_layout(pad=1.2)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _format_stats() -> str:
    """Return markdown for the live-stats panel."""
    global _obs, _total_reward, _env

    if _obs is None:
        return (
            "```\n"
            "  ██████████████████████████\n"
            "    CyberSec-SOC-OpenEnv\n"
            "    Press RESET to begin.\n"
            "  ██████████████████████████\n"
            "```"
        )

    stage        = _obs.attack_stage
    s_emoji, s_name = _STAGE.get(stage, ("❓", "Unknown"))
    stage_bar    = "█" * stage + "░" * (4 - stage)

    visible_threats = sum(1 for n in _obs.node_statuses if n["visible_compromise"])
    isolated_count  = sum(1 for n in _obs.node_statuses if n["is_isolated"])
    true_threats    = len(_env.state.true_compromised) if _env else 0

    if _obs.done and _obs.defender_wins:
        status_line = "### 🏆 DEFENDER WINS"
    elif _obs.done:
        status_line = "### 💀 ATTACKER WINS"
    else:
        active = true_threats > 0
        status_line = f"### ⚔️ {'Battle active' if active else 'Network secure'}"

    reward_sign = "▲" if _total_reward >= 0 else "▼"

    return f"""### 📡 Intel Feed

**Attack Stage:** `{stage}/4`
`[{stage_bar}]` {s_emoji} *{s_name}*

---

| Metric              | Value |
|---------------------|-------|
| ⏱ Timestep         | `{_obs.timestep}` |
| 💰 Biz Impact       | `{_obs.business_impact_score:.2f}` |
| 🔴 Active Threats   | `{true_threats}` |
| 👁 Detected         | `{visible_threats}` |
| 🔒 Isolated         | `{isolated_count}` |
| 🌐 Topology         | `{_obs.topology_type}` |

---

{status_line}

**Episode Reward:** `{reward_sign} {abs(_total_reward):.3f}`
"""


def _format_alerts() -> str:
    """Return SIEM-style alert text (newest first)."""
    global _obs
    if _obs is None:
        return "Waiting for first episode…"

    alerts = list(reversed((_obs.alerts or ["[t=0] No alerts yet."])[-12:]))
    lines  = []
    for a in alerts:
        upper = a.upper()
        if any(x in upper for x in ("EXFIL", "ATTACKER WINS", "STAGE 4", "SPREADS")):
            prefix = "🔴 CRITICAL  |"
        elif any(x in upper for x in ("STAGE 3", "STAGE 2", "COMPROMISE CONFIRMED")):
            prefix = "🟠 WARNING   |"
        elif any(x in upper for x in ("CONTAINED", "DEFENDER WINS", "FIREWALL")):
            prefix = "🟢 SUCCESS   |"
        elif any(x in upper for x in ("FALSE", "NOISE", "SUSPICIOUS")):
            prefix = "🟡 NOISE     |"
        elif "SCAN" in upper and "CLEAN" in upper:
            prefix = "🔵 INFO      |"
        else:
            prefix = "⚪ LOG       |"
        lines.append(f"{prefix} {a}")

    return "\n".join(lines)


def _format_score() -> str:
    """Return colour-coded HTML markdown score."""
    global _total_reward, _obs
    if _obs is None:
        return "### —"
    colour  = "#00ff88" if _total_reward >= 0 else "#ff4444"
    sign    = "+" if _total_reward >= 0 else ""
    value   = f"{sign}{_total_reward:.3f}"
    result  = ""
    if _obs.done:
        result = "<br><small>✅ Done</small>" if _obs.defender_wins else "<br><small>❌ Done</small>"
    return (
        f"<div style='text-align:center;padding:8px 0'>"
        f"<span style='font-size:2em;font-weight:900;font-family:monospace;"
        f"color:{colour};text-shadow:0 0 12px {colour}88'>{value}</span>"
        f"{result}</div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ACTION HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def _all_outputs():
    return _draw_network(), _format_stats(), _format_alerts(), _format_score()


def do_reset(task_level: str):
    global _env, _obs, _total_reward, _graph_pos
    with _lock:
        _env          = SOCEnvironment(task_level=task_level, seed=42)
        _obs          = _env.reset()
        _total_reward = 0.0
        _graph_pos    = None   # fresh spring-layout for new topology
    return _all_outputs()


def _do_step(action_type: str, node_id: int):
    global _env, _obs, _total_reward
    if _env is None or (_obs is not None and _obs.done):
        return _all_outputs()
    with _lock:
        action = SOCAction(action_type=action_type, target_node_id=int(node_id))
        _obs   = _env.step(action)
        if _obs.reward is not None:
            _total_reward += float(_obs.reward)
    return _all_outputs()


def do_scan(node_id):    return _do_step("scan",     int(node_id or 0))
def do_isolate(node_id): return _do_step("isolate",  int(node_id or 0))
def do_patch(node_id):   return _do_step("patch",    int(node_id or 0))
def do_firewall():       return _do_step("firewall", -1)
def do_nothing():        return _do_step("nothing",  -1)


# ══════════════════════════════════════════════════════════════════════════════
#  CSS & THEME
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
/* ── Root dark background ──────────────────────────────────────── */
.gradio-container, .gradio-container * { box-sizing: border-box; }
.gradio-container { background: #0b0f19 !important; color: #c8ddf0 !important; }

/* ── Panels ────────────────────────────────────────────────────── */
.panel-card {
    background: #111c2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    box-shadow: 0 0 20px rgba(0,120,255,0.08) !important;
}

/* ── All Gradio blocks inherit card style ───────────────────────── */
.block { background: #111c2e !important; border-color: #1e3a5f !important; }

/* ── Buttons — base ─────────────────────────────────────────────── */
button.lg { font-family: 'Courier New', monospace !important; font-weight: 700 !important; letter-spacing: 0.04em !important; border-radius: 6px !important; transition: all 0.15s !important; }

/* ── Coloured action buttons ────────────────────────────────────── */
.btn-reset  button.lg { background: #001a30 !important; border: 1px solid #00aaff !important; color: #00eeff !important; }
.btn-reset  button.lg:hover { background: #002a48 !important; box-shadow: 0 0 12px #00aaff55 !important; }
.btn-scan   button.lg { background: #001428 !important; border: 1px solid #0066cc !important; color: #55aaff !important; }
.btn-scan   button.lg:hover { background: #001f3d !important; }
.btn-iso    button.lg { background: #2a0e00 !important; border: 1px solid #cc4400 !important; color: #ff8844 !important; }
.btn-iso    button.lg:hover { background: #3d1500 !important; box-shadow: 0 0 10px #cc440044 !important; }
.btn-patch  button.lg { background: #00180a !important; border: 1px solid #00aa44 !important; color: #44ff88 !important; }
.btn-patch  button.lg:hover { background: #002210 !important; }
.btn-fw     button.lg { background: #160a28 !important; border: 1px solid #7744cc !important; color: #bb88ff !important; }
.btn-fw     button.lg:hover { background: #1e0f38 !important; box-shadow: 0 0 10px #7744cc44 !important; }
.btn-noop   button.lg { background: #141422 !important; border: 1px solid #334466 !important; color: #7788aa !important; }

/* ── Inputs ─────────────────────────────────────────────────────── */
input, textarea, select {
    background: #080f1a !important; color: #b0cce0 !important;
    border: 1px solid #1e3a5f !important; border-radius: 6px !important;
    font-family: 'Courier New', monospace !important;
}
input:focus, textarea:focus { border-color: #0088cc !important; outline: none !important; box-shadow: 0 0 8px #0088cc44 !important; }

/* ── Labels ─────────────────────────────────────────────────────── */
label span, .block-label { color: #4488aa !important; font-size: 0.78em !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; }

/* ── Markdown ───────────────────────────────────────────────────── */
.markdown-body, .prose { color: #b0cce0 !important; }
.markdown-body h3 { color: #00aaff !important; margin: 6px 0 4px !important; }
.markdown-body table { border-collapse: collapse !important; width: 100% !important; }
.markdown-body td, .markdown-body th { border: 1px solid #1e3a5f !important; padding: 4px 8px !important; color: #99bbd0 !important; font-family: 'Courier New', monospace !important; font-size: 0.82em !important; }
.markdown-body tr:nth-child(even) td { background: #0c1520 !important; }
.markdown-body hr { border-color: #1e3a5f !important; }
.markdown-body code { background: #0a1520 !important; color: #55ccff !important; padding: 1px 5px !important; border-radius: 3px !important; }

/* ── Alert feed textbox ─────────────────────────────────────────── */
.alert-box textarea { font-size: 0.79em !important; line-height: 1.65 !important; color: #99bbcc !important; }

/* ── Score box ──────────────────────────────────────────────────── */
.score-box { text-align: center !important; }
.score-box .output-html { display: flex !important; align-items: center !important; justify-content: center !important; min-height: 80px !important; }
"""

# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO BLOCKS LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    css=_CSS,
    title="CyberSec-SOC Dashboard",
    analytics_enabled=False,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Share Tech Mono"), "Courier New", "monospace"],
    ).set(
        body_background_fill="#0b0f19",
        block_background_fill="#111c2e",
        block_border_color="#1e3a5f",
        button_primary_background_fill="#001a30",
        button_primary_text_color="#00eeff",
        input_background_fill="#080f1a",
        body_text_color="#c8ddf0",
        block_label_text_color="#4488aa",
        block_title_text_color="#00aaff",
    ),
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="
        text-align:center; padding:14px 0 10px;
        border-bottom:1px solid #1e3a5f; margin-bottom:8px;
        background:linear-gradient(180deg,#0d1825 0%,#0b0f19 100%);
    ">
        <h1 style="
            color:#00ccff; font-size:1.75em; margin:0; letter-spacing:0.08em;
            font-family:'Courier New',monospace; font-weight:900;
            text-shadow:0 0 22px #00aaff88,0 0 4px #00ccff;
        ">🛡&nbsp; CyberSec-SOC Command Center</h1>
        <p style="color:#346080;font-size:0.84em;margin:4px 0 0;letter-spacing:0.05em;">
            AI-Powered Security Operations Center &nbsp;·&nbsp; Real-time Threat Defense Simulator
        </p>
    </div>
    """)

    # ── Main row: Graph | Controls | Stats ────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── LEFT: Network topology graph ──────────────────────────────────────
        with gr.Column(scale=6, min_width=400):
            graph_out = gr.Plot(label="Live Network Topology")

        # ── MIDDLE: Controls ──────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=190):
            gr.HTML("<div style='color:#00aaff;font-family:monospace;font-weight:700;"
                    "font-size:0.95em;padding-bottom:4px;border-bottom:1px solid #1e3a5f;"
                    "margin-bottom:8px'>🎮 &nbsp;CONTROL PANEL</div>")

            task_dd = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="medium",
                label="Task Level",
                info="easy=5 nodes · medium=10 · hard=20",
            )
            reset_btn = gr.Button(
                "⟳  RESET ENVIRONMENT",
                variant="primary",
                elem_classes=["btn-reset"],
                size="lg",
            )

            gr.HTML("<hr style='border-color:#1e3a5f;margin:10px 0'>")
            gr.HTML("<div style='color:#4488aa;font-family:monospace;font-size:0.8em;"
                    "margin-bottom:6px'>NODE ACTIONS</div>")

            node_inp = gr.Number(
                value=0, label="Target Node ID",
                precision=0, minimum=0, maximum=99,
            )

            with gr.Row():
                scan_btn = gr.Button("🔍 SCAN",    elem_classes=["btn-scan"],  size="lg")
                iso_btn  = gr.Button("🔒 ISOLATE", elem_classes=["btn-iso"],   size="lg")
            with gr.Row():
                patch_btn = gr.Button("🔧 PATCH",   elem_classes=["btn-patch"], size="lg")
                fw_btn    = gr.Button("🔥 FIREWALL",elem_classes=["btn-fw"],    size="lg")
            nothing_btn = gr.Button(
                "⏸  DO NOTHING", elem_classes=["btn-noop"], size="lg"
            )

            gr.HTML("""
            <div style='margin-top:10px;padding:8px;border:1px solid #1a3040;
                        border-radius:6px;background:#080f18'>
                <div style='color:#336655;font-size:0.72em;font-family:monospace;line-height:1.7'>
                💡 <b style='color:#447755'>Tip:</b> Scan → then Isolate confirmed<br>
                🔥 Firewall halves spread for 10 steps<br>
                ⚠ Isolating unscanned nodes costs BIZ
                </div>
            </div>
            """)

        # ── RIGHT: Live stats ─────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=210):
            stats_md = gr.Markdown("**Press RESET to start.**")

    # ── Bottom row: SIEM feed | Score ─────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=5):
            alerts_box = gr.Textbox(
                label="🚨 SIEM Alert Feed  (newest first)",
                lines=9, max_lines=9,
                interactive=False,
                elem_classes=["alert-box"],
                placeholder="Security alerts will stream here after RESET…",
            )

        with gr.Column(scale=1, min_width=170):
            gr.HTML("""
            <div style='color:#4488aa;font-family:monospace;font-size:0.78em;
                        text-transform:uppercase;letter-spacing:0.05em;
                        padding-bottom:5px;border-bottom:1px solid #1e3a5f;
                        margin-bottom:8px'>
                📈 Episode Reward
            </div>
            """)
            score_html = gr.HTML(
                "<div style='text-align:center;padding:18px 0;"
                "font-size:2em;font-weight:900;font-family:monospace;color:#446688'>—</div>",
                elem_classes=["score-box"],
            )
            gr.HTML("""
            <div style='text-align:center;color:#2a4455;font-size:0.72em;
                        font-family:monospace;margin-top:6px;line-height:1.6'>
                ▲ Positive = defending<br>▼ Negative = losing<br>
                <span style='color:#335533'>✔</span> Win: all threats contained<br>
                <span style='color:#553333'>✖</span> Lose: exfiltration completes
            </div>
            """)

    # ── Footer status bar ──────────────────────────────────────────────────────
    gr.HTML("""
    <div style='
        margin-top:12px;padding:6px 14px;
        border-top:1px solid #1a3040;
        display:flex;gap:24px;align-items:center;
        background:#080f18;border-radius:0 0 8px 8px;
    '>
        <span style='color:#1e4a60;font-family:monospace;font-size:0.72em'>
            🌐 ENV: <span style='color:#2a6a8a'>localhost:8000</span>
        </span>
        <span style='color:#1e4a60;font-family:monospace;font-size:0.72em'>
            📡 PROTOCOL: <span style='color:#2a6a8a'>Direct (in-process)</span>
        </span>
        <span style='color:#1e4a60;font-family:monospace;font-size:0.72em'>
            🛡 CyberSec-SOC-OpenEnv · OpenEnv v0.2.3
        </span>
    </div>
    """)

    # ── Event wiring ───────────────────────────────────────────────────────────
    _outs = [graph_out, stats_md, alerts_box, score_html]

    reset_btn.click(fn=do_reset,    inputs=[task_dd],  outputs=_outs)
    scan_btn.click( fn=do_scan,     inputs=[node_inp], outputs=_outs)
    iso_btn.click(  fn=do_isolate,  inputs=[node_inp], outputs=_outs)
    patch_btn.click(fn=do_patch,    inputs=[node_inp], outputs=_outs)
    fw_btn.click(   fn=do_firewall, inputs=[],         outputs=_outs)
    nothing_btn.click(fn=do_nothing,inputs=[],         outputs=_outs)
