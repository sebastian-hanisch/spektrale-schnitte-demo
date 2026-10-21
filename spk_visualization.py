"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import spk_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _lines(inst, pairs):
    xs, ys = [], []
    for u, v in pairs:
        xs += [float(inst.xy[u, 0]), float(inst.xy[v, 0]), None]
        ys += [float(inst.xy[u, 1]), float(inst.xy[v, 1]), None]
    return xs, ys


def _map_layout(fig, inst, height=470):
    xs, ys = inst.xy[:, 0], inst.xy[:, 1]
    pad = 0.25 * max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    fig.update_xaxes(visible=False, range=[float(xs.min()) - pad, float(xs.max()) + pad])
    fig.update_yaxes(visible=False, range=[float(ys.min()) - pad, float(ys.max()) + pad])
    return _base(fig, height)


def _edge_traces(fig, inst, cross):
    cross = set(cross)
    rest = [(u, v) for u, v in inst.pairs if (u, v) not in cross]
    xs, ys = _lines(inst, rest)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_EDGE, width=1.2), hoverinfo="skip", showlegend=False))
    xs, ys = _lines(inst, sorted(cross))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_CUT, width=2.6, dash="dot"), hoverinfo="skip", name=f"Schnitt ({len(cross)})"))


def _marker_text(inst, show_text):
    return [inst.name(v) for v in range(inst.n)] if show_text else None


def build_partition_map(inst, side_set, height=470):
    """Karte mit einer Menge S (grün), ihrem Rest (blau) und den Leitungen dazwischen (rot, gestrichelt)."""
    side_set = set(side_set)
    cross = [(u, v) for u, v in inst.pairs if (u in side_set) != (v in side_set)]
    fig = go.Figure()
    _edge_traces(fig, inst, cross)
    show_text = inst.n <= 60
    size = 22 if show_text else 11
    for members, color, name in ((sorted(side_set), C.COLOR_SIDE_A, "Menge S"), ([v for v in range(inst.n) if v not in side_set], C.COLOR_SIDE_B, "Rest")):
        fig.add_trace(go.Scatter(
            x=[float(inst.xy[v, 0]) for v in members], y=[float(inst.xy[v, 1]) for v in members], mode="markers+text" if show_text else "markers",
            marker=dict(size=size, color=color, line=dict(color="white", width=1)), text=[inst.name(v) for v in members] if show_text else None, textfont=dict(size=10, color="#ffffff"),
            textposition="middle center", hovertext=[f"Standort {inst.name(v)}" for v in members], hoverinfo="text", name=name))
    return _map_layout(fig, inst, height)


def build_fiedler_map(inst, x, side_set, height=470):
    """Karte, die Standorte nach dem Fiedler-Vektor gefärbt; der Schnitt des besten Anfangs ist rot gestrichelt."""
    side_set = set(side_set)
    cross = [(u, v) for u, v in inst.pairs if (u in side_set) != (v in side_set)]
    fig = go.Figure()
    _edge_traces(fig, inst, cross)
    show_text = inst.n <= 60
    fig.add_trace(go.Scatter(
        x=[float(inst.xy[v, 0]) for v in range(inst.n)], y=[float(inst.xy[v, 1]) for v in range(inst.n)], mode="markers+text" if show_text else "markers",
        marker=dict(size=22 if show_text else 11, color=list(x), colorscale="RdBu", cmid=0.0, line=dict(color="white", width=1), colorbar=dict(title="Fiedler-Vektor", thickness=12, len=0.7)),
        text=_marker_text(inst, show_text), textfont=dict(size=10, color="#222222"), textposition="middle center",
        hovertext=[f"Standort {inst.name(v)}: {x[v]:.3f}" for v in range(inst.n)], hoverinfo="text", showlegend=False))
    return _map_layout(fig, inst, height)


def build_sorted_values(a):
    """Der Fiedler-Vektor, der Größe nach sortiert: Lücken und Stufen zeigen, wo der Sweep schneidet; senkrecht der beste Anfang."""
    sw, inst = a.sw, a.inst
    xs = list(a.spec.x)
    order = sw.order
    vals = [xs[v] for v in order]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, a.n + 1)), y=vals, mode="markers+lines", marker=dict(size=7, color=[C.COLOR_SIDE_A if i < a.best_k else C.COLOR_SIDE_B for i in range(a.n)]),
                             line=dict(color="rgba(150,150,150,0.5)", width=1), hovertext=[f"Standort {inst.name(v)}" for v in order], hoverinfo="text+y", showlegend=False))
    fig.add_vline(x=a.best_k + 0.5, line=dict(color=C.COLOR_CUT, dash="dot", width=2))
    fig.update_xaxes(title_text="Rang nach dem Fiedler-Vektor", range=[0.5, a.n + 0.5])
    fig.update_yaxes(title_text="Fiedler-Vektor")
    return _base(fig, 300)


def build_sweep_curve(a):
    """Wert des gewählten Maßes für jeden Anfang der Sortierung; Marke: bester Anfang. Bei der Konduktanz zusätzlich die Schranken von Cheeger und h."""
    sw, s = a.sw, a.settings
    ks = list(range(1, a.n))
    series = sw.series(s.measure)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=list(series), mode="lines+markers", line=dict(color=C.COLOR_BAR, width=2), marker=dict(size=5), name=C.MEASURE_SHORT[s.measure], hovertemplate="Anfang %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[a.best_k], y=[series[a.best_k - 1]], mode="markers", marker=dict(size=13, color=C.COLOR_CUT, symbol="diamond"), name="bester Anfang"))
    if a.exact is not None:
        fig.add_hline(y=a.optimum, line=dict(color=C.COLOR_BAR3, dash="dash", width=2), annotation_text="Optimum (Aufzählung)", annotation_position="top left")
    if s.measure == "cond":
        lo, hi = a.cheeger
        fig.add_hline(y=lo, line=dict(color="#8a9a9c", dash="dot", width=1.5), annotation_text="λ₂/2", annotation_position="bottom right")
        fig.add_hline(y=hi, line=dict(color="#8a9a9c", dash="dot", width=1.5), annotation_text="√(2 λ₂)", annotation_position="top right")
    fig.add_vline(x=a.median_k, line=dict(color=C.COLOR_BAR2, dash="dot", width=1.5), annotation_text="Hälfte", annotation_position="top")
    fig.update_xaxes(title_text="Länge des Anfangs der Sortierung (Standorte in S)", dtick=max(1, a.n // 12))
    fig.update_yaxes(title_text=C.MEASURE_NAMES[s.measure], rangemode="tozero")
    return _base(fig, 340)


def build_cheeger(a):
    """Die Schranken von Cheeger, die kleinste Konduktanz (Aufzählung) und der Sweep-Cut auf einer Zahlengeraden (Balken von 0 bis zum Wert)."""
    lo, hi = a.cheeger
    sweep_cond = min(a.sweeps["norm"].cond)
    labels = ["untere Schranke λ₂/2", "kleinste Konduktanz h (Aufzählung)", "Sweep-Cut (normalisiert)", "obere Schranke √(2 λ₂)"]
    vals = [lo, a.h, sweep_cond, hi]
    colors = ["#8a9a9c", C.COLOR_BAR3, C.COLOR_BAR, "#8a9a9c"]
    keep = [i for i, v in enumerate(vals) if v is not None]
    fig = go.Figure(go.Bar(x=[vals[i] for i in keep], y=[labels[i] for i in keep], orientation="h", marker_color=[colors[i] for i in keep], hovertemplate="%{y}: %{x:.3f}<extra></extra>", showlegend=False))
    fig.update_xaxes(title_text="Konduktanz", rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 240)


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------------


def build_exact(rows):
    ys = [round(100 * (r["checks"] - r["wrong"]) / r["checks"]) if r["checks"] else 0 for r in rows]
    fig = go.Figure(go.Bar(x=[r["label"] for r in rows], y=ys, marker_color=C.COLOR_BAR, showlegend=False, hovertemplate="%{x}: %{y} %<extra></extra>"))
    fig.update_yaxes(title_text="Prüfungen ohne Abweichung (%)", range=[0, 105])
    return _base(fig)


def build_objective(rows):
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    for key, name, color in (("cut_all_trivial", "kleinster Schnitt", C.COLOR_BAR), ("ratio_trivial", "bester RatioCut", C.COLOR_BAR2), ("ncut_trivial", "bester Ncut", C.COLOR_BAR3), ("cond_trivial", "kleinste Konduktanz", C.COLOR_LOCK)):
        fig.add_trace(go.Bar(x=labels, y=[round(100 * r[key] / r["nets"]) for r in rows], name=name, marker_color=color, hovertemplate="%{x}: %{y} %<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Netze, in denen der beste Schnitt nur ein Standort ist (%)", range=[0, 105])
    return _base(fig, 380)


def build_cheeger_rows(rows):
    labels = [r["label"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("h geteilt durch die untere Schranke λ₂/2", "h geteilt durch die obere Schranke √(2 λ₂)"), horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=labels, y=[r["lo_max"] - r["lo_min"] for r in rows], base=[r["lo_min"] for r in rows], marker_color=C.COLOR_BAR, showlegend=False,
                         hovertemplate="%{x}: von %{base:.2f} bis %{y:.2f} über dem Anfang<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=[r["hi_max"] - r["hi_min"] for r in rows], base=[r["hi_min"] for r in rows], marker_color=C.COLOR_BAR2, showlegend=False,
                         hovertemplate="%{x}: von %{base:.2f} bis %{y:.2f} über dem Anfang<extra></extra>"), row=1, col=2)
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_CUT, dash="dash", width=2), row=1, col=1)
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_CUT, dash="dash", width=2), row=1, col=2)
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(showticklabels=False)
    return _base(fig, 330)


def build_sweep_hits(rows):
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    for key, name, color in (("hit_cond", "Konduktanz, normalisiert", C.COLOR_BAR), ("hit_cond_plain", "Konduktanz, unnormalisiert", "rgba(76,120,168,0.45)"), ("hit_ncut", "Ncut, normalisiert", C.COLOR_BAR3),
                             ("hit_ratio_plain", "RatioCut, unnormalisiert", C.COLOR_BAR2)):
        fig.add_trace(go.Bar(x=labels, y=[round(100 * r[key] / r["nets"]) for r in rows], name=name, marker_color=color, hovertemplate="%{x}: %{y} %<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Netze, in denen der Sweep das Optimum trifft (%)", range=[0, 105])
    return _base(fig, 380)


def build_big(rows):
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[round(r["sum_phi_sweep"] / r["nets"], 3) for r in rows], name="bester Sweep-Anfang", marker_color=C.COLOR_BAR3, hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=[round(r["sum_phi_median"] / r["nets"], 3) for r in rows], name="Halbierung am Median", marker_color=C.COLOR_BAR2, hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Konduktanz (Mittel über die Netze)", rangemode="tozero")
    return _base(fig)
