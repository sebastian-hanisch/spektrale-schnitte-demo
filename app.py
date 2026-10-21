"""Spektrale Schnitte: Fiedler-Vektor, Sweep-Cut und Cheeger - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der Reihe „Schnitte und Zusammenhang“ (gebaut als drittes nach Stück 5): Warum kann ein Eigenvektor einen Schnitt finden, und wie gut? Fiedler-Vektor der Laplace-Matrix, Sweep-Cut,
die Zielfunktionen (kleinster Schnitt, RatioCut, Ncut, Konduktanz) nebeneinander und die Schranken von Cheeger.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import spk_algorithm as A
import spk_constants as C
import spk_evaluation as ev
import spk_experiments as ex
from spk_layout import dunkles_design, kopf_css
from spk_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from spk_visualization import (
    build_big,
    build_cheeger,
    build_cheeger_rows,
    build_exact,
    build_fiedler_map,
    build_objective,
    build_partition_map,
    build_sorted_values,
    build_sweep_curve,
    build_sweep_hits,
)

st.set_page_config(page_title="Spektrale Schnitte: Fiedler-Vektor, Sweep-Cut und Cheeger – Sebastian Hanisch", layout="wide")

_i, _dec, _v = ev._i, ev._dec, ev._v


def _show(verdict):
    kind, text = verdict
    {"success": st.success, "info": st.info, "warning": st.warning}[kind](text)


@st.cache_data(show_spinner=False, max_entries=8)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _exactness():
    return ex.exactness_rows()


@st.cache_data(show_spinner=False)
def _objective():
    return ex.objective_rows()


@st.cache_data(show_spinner=False)
def _cheeger():
    return ex.cheeger_rows()


@st.cache_data(show_spinner=False)
def _sweep():
    return ex.sweep_rows()


@st.cache_data(show_spinner=False)
def _big():
    return ex.big_rows()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in spk_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔭 Spektrale Schnitte: Fiedler-Vektor, Sweep-Cut und Cheeger")
st.markdown(
    """
**Warum kann ein Eigenvektor einen Schnitt finden, und wie gut?** Der kleinste Schnitt eines Netzes trennt meist nur **einen einzelnen Standort** ab. Wer eine **sinnvolle Teilung** sucht, braucht ein Maß, das große Teile
belohnt: **RatioCut** teilt durch die Zahl der Standorte, **Ncut** und die **Konduktanz** durch das Volumen (die Summe der Grade). Diese Aufgaben sind schwer; die **spektrale** Antwort ist ein Eigenvektor: Der Vektor zum zweitkleinsten
Eigenwert der Laplace-Matrix (der **Fiedler-Vektor**) ordnet die Standorte, und der **Sweep-Cut** schneidet an der besten Stelle dieser Ordnung. Die **Ungleichung von Cheeger** sagt, wie gut das höchstens sein kann.

Auffällig: Der kleinste Schnitt und die ausgewogenen Maße wählen **verschiedene** Schnitte; die Schranken von Cheeger gelten ausnahmslos, sind aber **nie eng**; und wo λ₂ mehrfach ist (Gitter), ist der Fiedler-Vektor nicht eindeutig.

Das ist das siebte Stück der Reihe **Schnitte und Zusammenhang**; den kleinsten Schnitt zeigt **globaler-schnitt-demo**, die Halbierung durch Tausch **bisektion-demo**. Als Verfahren der Gruppenbildung (k-Means auf der Einbettung)
steht der Eigenvektor in **spectral-demo**, die Modularität in **leiden-demo**; hier geht es um die **Schnittsicht**.
Weiter unten: die Experimente **🔬**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So findet ein Eigenvektor einen Schnitt", expanded=True):
    st.markdown(
        """
1. **Schnittmaße:** Für eine Menge S ist c(S) die Zahl der Leitungen zwischen S und dem Rest. **RatioCut** = c / |S| + c / (n − |S|), **Ncut** = c / vol(S) + c / vol(Rest), **Konduktanz** φ(S) = c / min(vol(S), vol(Rest)); die kleinste Konduktanz heißt **Cheeger-Konstante** h.
2. **Laplace-Matrix:** L = D − A (unnormalisiert, passt zu RatioCut) bzw. L = I − D^(−½) A D^(−½) (normalisiert, passt zu Ncut und Konduktanz). Der Eigenwert 0 gehört zum trivialen Vektor; der Vektor zum **zweitkleinsten Eigenwert λ₂** ist der **Fiedler-Vektor**.
3. **Sweep-Cut:** Die Standorte werden nach dem Fiedler-Vektor (bei der normalisierten Matrix: D^(−½) mal Eigenvektor) sortiert; jeder **Anfang** der Sortierung ist ein Schnitt. Man nimmt den mit dem besten Wert.
4. **Cheeger:** Für die normalisierte Matrix gilt λ₂ / 2 ≤ h ≤ √(2 λ₂), und der Sweep-Cut erreicht einen Wert höchstens √(2 λ₂).

| Begriff | Bedeutung |
|---|---|
| **Fiedler-Vektor** | Eigenvektor zum zweitkleinsten Eigenwert der Laplace-Matrix; sein Vorzeichen und seine Größe ordnen die Standorte |
| **Sweep-Cut** | der beste Anfang der nach dem Fiedler-Vektor sortierten Standorte |
| **Volumen** | Summe der Grade der Standorte einer Menge (jede Leitung zählt an beiden Enden) |
| **Konduktanz φ(S)** | Schnitt geteilt durch das kleinere der beiden Volumen |
| **Cheeger-Konstante h** | kleinste Konduktanz über alle Schnitte (bei bis zu 18 Standorten durch Aufzählung) |
| **Vielfachheit von λ₂** | Zahl der Eigenwerte gleich λ₂: ab zwei ist der Fiedler-Vektor nicht eindeutig |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    kind = st.selectbox("Art des Netzes", list(C.KINDS), format_func=C.KIND_NAMES.get, key="kind_select", help=C.KIND_HELP.get(st.session_state.get("kind_select", C.DEFAULT_KIND), ""))
    n = st.slider("Standorte n", *bounds("n_slider"), key="n_slider", help=f"Wie viele Standorte das Netz hat (das Gitter rundet auf ein Quadrat). Die Aufzählung aller Schnitte läuft bis {C.BRUTE_N} Standorte.")
    deg = st.slider("Mittlerer Grad", *bounds("deg_slider"), key="deg_slider", step=0.5, help="Leitungen je Standort im Mittel; wirkt beim Zufallsnetz und innerhalb der Regionen des Verbundnetzes.")
    links = st.slider("Leitungen zwischen je zwei Regionen", *bounds("links_slider"), key="links_slider", help="Wirkt nur beim Verbundnetz.")
    seed = st.number_input("Zufalls-Seed des Netzes", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Netz erzeugen", width="stretch", on_click=randomize_seed)
    st.caption(C.KIND_REGLER[kind])
    st.markdown("**Verfahren**")
    lap = st.selectbox("Matrix des Eigenvektors", list(C.LAPS), format_func=C.LAP_NAMES.get, key="lap_select", help="Welche Laplace-Matrix den Fiedler-Vektor liefert; beide werden gerechnet, die Cheeger-Schranken gelten für die normalisierte.")
    measure = st.selectbox("Maß der Schnittwahl", list(C.MEASURES), format_func=C.MEASURE_NAMES.get, key="measure_select", help="Welcher Anfang der Sortierung gewählt wird: der mit dem kleinsten Wert dieses Maßes.")

sync_query_params({"kind_select": kind, "n_slider": int(n), "deg_slider": float(deg), "links_slider": int(links), "seed_input": int(seed), "lap_select": lap, "measure_select": measure})

settings = ev.Settings(kind, int(n), float(deg), int(links), int(seed), lap, measure)
with st.spinner("Rechne das Spektrum und die Schnitte..."):
    a = _analysis(settings)
problems = ev.validate(a)
for p in problems:
    st.error(f"Prüfung fehlgeschlagen: {p}")
signature = f"{settings.kind}_{settings.n}_{settings.deg}_{settings.links}_{settings.seed}_{settings.lap}_{settings.measure}"
inst = a.inst
sw = a.sw
mshort = C.MEASURE_SHORT[measure]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"{C.KIND_NAMES[kind]} mit {_i(a.n)} Standorten und {_i(a.m)} Leitungen. Standorte sind ab 1 nummeriert (das Lehrbeispiel hat Buchstaben).")
side_c, side_r, side_nc, side_phi = a.side_measures
lo, hi = a.cheeger
m1, m2, m3, m4 = st.columns(4)
m1.metric("Kleinster Schnitt ohne Balance", _i(a.mincut), delta="ein einzelner Standort" if a.mincut_trivial else "echte Teilung", delta_color="off")
m2.metric(f"Sweep-Cut: {mshort}", _v(a.sweep_value), delta=f"{_i(a.best_k)} gegen {_i(a.n - a.best_k)} Standorte, Schnitt {_i(side_c)}", delta_color="off")
if a.exact:
    opt = a.optimum
    m3.metric(f"Optimum ({mshort})", _v(opt), delta="trifft das Optimum" if abs(a.sweep_value - opt) <= C.TOL else f"Sweep {_dec(a.sweep_value / opt)}-fach", delta_color="off" if abs(a.sweep_value - opt) <= C.TOL else "inverse")
else:
    m3.metric(f"Optimum ({mshort})", "keine Aufzählung", delta=f"über {C.BRUTE_N} Standorte", delta_color="off")
m4.metric("λ₂ (normalisiert)", _v(a.spectra["norm"].lam2), delta=f"h zwischen {_v(lo)} und {_v(hi)}", delta_color="off")
_show(ev.verdict_question(a))

tab_sweep, tab_goal, tab_cheeger = st.tabs(["🧭 Eigenvektor und Sweep", "🎯 Welches Ziel?", "📐 Cheeger"])

# --- Tab 1: Eigenvektor und Sweep ----------------------------------------------------------------------------------------------------------

with tab_sweep:
    k = st.slider("Länge des Anfangs der Sortierung (Standorte in S)", 1, a.n - 1, a.best_k, key=f"k_{signature}", help="Zeigt den Schnitt, der aus den ersten k Standorten der Sortierung nach dem Fiedler-Vektor besteht; vorbelegt ist der beste.")
    prefix = sw.prefix(int(k))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fiedler-Vektor auf der Karte**")
        st.plotly_chart(build_fiedler_map(inst, a.spec.x, prefix), width="stretch", key=f"fiedler_map_{signature}_{int(k)}")
        st.caption("Rot und blau: Vorzeichen und Größe des Fiedler-Vektors. Gestrichelt rot: der Schnitt des gewählten Anfangs.")
    with c2:
        st.markdown("**Der Schnitt des gewählten Anfangs**")
        st.plotly_chart(build_partition_map(inst, prefix), width="stretch", key=f"partition_map_{signature}_{int(k)}")
        st.caption("Grün: die ersten k Standorte der Sortierung (Menge S), blau: der Rest.")
    kk = int(k) - 1
    rowk = pd.DataFrame([{"Standorte in S": _i(int(k)), "Schnitt": _i(sw.cut[kk]), "Volumen von S": _i(sw.vol[kk]), "RatioCut": _v(sw.ratio[kk]), "Ncut": _v(sw.ncut[kk]), "Konduktanz": _v(sw.cond[kk])}])
    st.dataframe(rowk, hide_index=True, width="stretch")
    st.markdown("**Der Fiedler-Vektor, sortiert**")
    st.plotly_chart(build_sorted_values(a), width="stretch", key=f"sorted_{signature}")
    st.caption("Jeder Punkt ein Standort in der Reihenfolge der Sortierung. Stufen und Lücken zeigen natürliche Schnittstellen; senkrecht rot der beste Anfang für das gewählte Maß.")
    st.markdown(f"**{C.MEASURE_NAMES[measure]} für jeden Anfang der Sortierung**")
    st.plotly_chart(build_sweep_curve(a), width="stretch", key=f"sweep_curve_{signature}")
    st.caption("Der Sweep prüft nur n − 1 Schnitte statt aller 2^(n−1) − 1. Gelb gestrichelt: die Halbierung am Median. Bei der Konduktanz zeigen graue Linien die Schranken von Cheeger.")
    if a.spec.multiplicity > 1:
        st.warning(f"λ₂ ist hier {_i(a.spec.multiplicity)}-fach: Der Eigenraum ist mehrdimensional, der Fiedler-Vektor ist nicht eindeutig. Die Demo nimmt die Projektion eines festen Hilfsvektors auf den Eigenraum; ein anderer Vektor "
                   f"im Eigenraum liefert eine andere Sortierung und einen anderen Sweep.")

# --- Tab 2: Welches Ziel? ------------------------------------------------------------------------------------------------------------------

with tab_goal:
    st.markdown("**Vier Ziele, vier Schnitte**")
    sides, source = {}, {}
    for key in C.OBJECTIVES:
        if key == "cut":
            sides[key], source[key] = tuple(a.mincut_side), "Stoer–Wagner"
        elif a.exact:
            sides[key], source[key] = a.exact.opt[key].side, "Aufzählung"
        else:
            kmeasure = sw.best(key)
            sides[key], source[key] = sw.prefix(kmeasure), "bester Sweep-Anfang"
    rows = []
    for key in C.OBJECTIVES:
        s = set(sides[key])
        c_, r_, nc_, co_ = A.measures(inst.adj, s)
        size = min(len(s), a.n - len(s))
        rows.append({"Ziel": C.OBJECTIVE_NAMES[key], "Herkunft": source[key], "Schnitt": _i(c_), "Standorte (kleinere Seite)": _i(size) + (" (einzelner Standort)" if size == 1 else ""), "RatioCut": _v(r_), "Ncut": _v(nc_), "Konduktanz": _v(co_)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    g1, g2 = st.columns(2)
    for i, key in enumerate(C.OBJECTIVES):
        col = g1 if i % 2 == 0 else g2
        with col:
            st.markdown(f"**{C.OBJECTIVE_NAMES[key]}**")
            st.plotly_chart(build_partition_map(inst, sides[key], height=340), width="stretch", key=f"goal_{key}_{signature}")
    if a.exact:
        o = a.exact.opt
        st.caption(f"Es gibt {_i(a.exact.total)} Schnitte. Die Zahl der besten: kleinster Schnitt {_i(o['cut'].count)}, RatioCut {_i(o['ratio'].count)}, Ncut {_i(o['ncut'].count)}, Konduktanz {_i(o['cond'].count)}. "
                   f"Ncut und RatioCut haben {'dieselben' if a.exact.same('ncut', 'ratio') else 'verschiedene'} besten Schnitte, Ncut und Konduktanz {'dieselben' if a.exact.same('ncut', 'cond') else 'verschiedene'}.")
    else:
        st.caption(f"Über {C.BRUTE_N} Standorten gibt es keine Aufzählung: Die drei ausgewogenen Maße zeigen den besten Anfang des Sweeps ({C.LAP_NAMES[lap]}), der kleinste Schnitt kommt von Stoer–Wagner.")

# --- Tab 3: Cheeger ------------------------------------------------------------------------------------------------------------------------

with tab_cheeger:
    st.markdown("**Die Schranken von Cheeger**")
    if a.exact:
        st.plotly_chart(build_cheeger(a), width="stretch", key=f"cheeger_{signature}")
        st.caption("Die kleinste Konduktanz h liegt zwischen den beiden grauen Balken; der Sweep-Cut mit der normalisierten Matrix liegt zwischen h und der oberen Schranke.")
    else:
        st.info(f"Über {C.BRUTE_N} Standorten fehlt h (keine Aufzählung): Es bleiben λ₂ / 2 = {_v(lo)} als untere Schranke, √(2 λ₂) = {_v(hi)} als obere und der Sweep-Cut mit {_v(min(a.sweeps['norm'].cond))}.")
    ev_rows = []
    for lap_key in C.LAPS:
        sp = a.spectra[lap_key]
        ev_rows.append({"Matrix": C.LAP_NAMES[lap_key], "λ₂": _v(sp.lam2), "Vielfachheit von λ₂": _i(sp.multiplicity), "Eigenwerte 1 bis 6": ", ".join(_v(t) for t in sp.values[:6])})
    st.dataframe(pd.DataFrame(ev_rows), hide_index=True, width="stretch")
    st.caption("Die Schranken gelten nur für die normalisierte Matrix; der kleinste Eigenwert ist immer 0 (das Netz ist zusammenhängend, also nur einmal).")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🔬 Experimente über feste Netze")
st.caption(f"Je Netztyp {len(C.SWEEP_SEEDS)} feste Netze mit {C.EX_N} Standorten (7 Netztypen), feste Zufalls-Seeds: nie ein einzelner Zufalls-Seed als Befund.")

st.subheader("🔬 Experiment 1: Stimmen Spektrum, Sweep und Aufzählung überein?")
rows_x = _exactness()
st.caption(f"{len(rows_x)} Netztypen mit je {len(C.SWEEP_SEEDS)} festen Netzen mit {C.EX_N} Standorten; geprüft werden Eigenwertgleichung, Schranken von Cheeger, jeder Anfang des Sweeps und jeder beste Schnitt der Aufzählung.")
st.plotly_chart(build_exact(rows_x), width="stretch", key="ex_exact")
_show(ev.verdict_exact(rows_x))

st.subheader("🔬 Experiment 2: Welchen Schnitt wählt welches Ziel?")
rows_o = _objective()
st.caption(f"Anteil der Netze ({C.EX_N} Standorte, je Netztyp {len(C.SWEEP_SEEDS)}), in denen der beste Schnitt nur einen einzelnen Standort abtrennt, nach Ziel.")
st.plotly_chart(build_objective(rows_o), width="stretch", key="ex_objective")
_show(ev.verdict_objective(rows_o))

st.subheader("🔬 Experiment 3: Wie eng sind die Schranken von Cheeger?")
rows_c = _cheeger()
st.caption("Verhältnis der kleinsten Konduktanz h zur unteren und zur oberen Schranke, kleinster und größter Wert je Netztyp (die rote Linie bei 1 wäre eine scharfe Schranke).")
st.plotly_chart(build_cheeger_rows(rows_c), width="stretch", key="ex_cheeger")
_show(ev.verdict_cheeger(rows_c))

st.subheader("🔬 Experiment 4: Wie oft trifft der Sweep-Cut das Optimum?")
rows_s = _sweep()
st.caption(f"Anteil der Netze ({C.EX_N} Standorte, je Netztyp {len(C.SWEEP_SEEDS)}), in denen der beste Anfang der Sortierung das Optimum der Aufzählung trifft, nach Matrix und Maß.")
st.plotly_chart(build_sweep_hits(rows_s), width="stretch", key="ex_sweep")
_show(ev.verdict_sweep(rows_s))

st.subheader("🔬 Experiment 5: Sweep-Cut gegen Halbierung am Median im großen Netz")
rows_b = _big()
st.caption(f"Bei {C.EX_BIG_N} Standorten (keine Aufzählung), {C.EX_BIG_SEEDS} Netze je Typ: Konduktanz des besten Sweep-Anfangs gegen die der Hälfte der Sortierung (normalisierte Matrix).")
st.plotly_chart(build_big(rows_b), width="stretch", key="ex_big")
_show(ev.verdict_big(rows_b))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein Eigenvektor genügt** | Der Fiedler-Vektor sortiert gut, wenn das Netz einen deutlichen Engpass hat; bei mehrfachem λ₂ (Gitter) ist er nicht eindeutig, bei vielen kleinen Engpässen fasst eine Ordnung nicht alle. | - |
| **Schranken sind nicht eng** | Die Ungleichung von Cheeger sagt nur, dass der Sweep nie schlechter als √(2 λ₂) ist; sie sagt nicht, dass er das Optimum trifft (Experiment 3 und 4). | - |
| **Gleiche Leitungen** | Alle Leitungen zählen gleich; mit Gewichten wären Volumen und Schnitt gewichtete Summen. | globaler-schnitt-demo |
| **Zwei Teile** | Mehr als zwei Teile brauchen mehrere Eigenvektoren (k-Means auf der Einbettung) oder wiederholtes Teilen. | spectral-demo |
| **Zusammenhang der Teile** | Weder der Sweep noch die Maße verlangen, dass beide Seiten zusammenhängen; die Anfänge der Sortierung können zerfallen. | - |
| **Kein dünnster Schnitt in Polynomialzeit exakt** | Der kleinste Wert eines der Maße ist NP-schwer; Verfahren mit Garantie (Arora–Rao–Vazirani, Expander-Zerlegung) sind nicht gebaut und stehen nur als Einordnung hier. | - |
| **Synthetische Netze, bis 120 Standorte** | Reale Netze haben andere Strukturen; dichte Eigenzerlegung genügt hier, große Netze brauchen Lanczos-Verfahren. Die Aufzählung läuft bis 18 Standorte. | - |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben** ein einfacher zusammenhängender Graph $G = (V, E)$ mit $n = |V|$ Standorten, Adjazenzmatrix $A$, Grad $d(v)$, Gradmatrix $D$ und $\operatorname{vol}(S) = \sum_{v \in S} d(v)$, also $\operatorname{vol}(V) = 2|E|$. Für $\emptyset \ne S \subsetneq V$ sei $c(S)$ die Zahl der Leitungen zwischen $S$ und $\bar S = V \setminus S$.

**Schnittmaße.**

$$\operatorname{RatioCut}(S) = \frac{c(S)}{|S|} + \frac{c(S)}{|\bar S|}$$

$$\operatorname{Ncut}(S) = \frac{c(S)}{\operatorname{vol}(S)} + \frac{c(S)}{\operatorname{vol}(\bar S)}$$

$$\varphi(S) = \frac{c(S)}{\min(\operatorname{vol}(S), \operatorname{vol}(\bar S))}, \qquad h = \min_S \varphi(S)$$

**Laplace-Matrizen.** Unnormalisiert $L = D - A$, normalisiert $\mathcal{L} = I - D^{-1/2} A D^{-1/2}$. Beide sind symmetrisch und positiv semidefinit; für zusammenhängende Graphen ist der Eigenwert 0 einfach, mit dem Eigenvektor $\mathbf{1}$ (für $L$) bzw. $D^{1/2}\mathbf{1}$ (für $\mathcal{L}$), und $\lambda_2 > 0$.

**Fiedler-Vektor.** Zu $\lambda_2$ gehört ein Eigenvektor $v$ senkrecht zum trivialen. Für $\mathcal{L}$ sortiert man nach $x = D^{-1/2} v$, für $L$ nach $x = v$. Die Relaxation von RatioCut führt auf $L$, die von Ncut auf $\mathcal{L}$:

$$\lambda_2(L) = \min_{x \perp \mathbf{1}} \frac{x^\top L x}{x^\top x}, \qquad x^\top L x = \sum_{\{u,v\} \in E} (x_u - x_v)^2$$

**Sweep-Cut.** Mit der Sortierung $x_{\pi(1)} \le \dots \le x_{\pi(n)}$ sind $S_k = \{\pi(1), \dots, \pi(k)\}$, $k = 1, \dots, n-1$, die $n-1$ geprüften Schnitte; gewählt wird der mit dem besten Wert des Maßes.

**Ungleichung von Cheeger** (normalisierte Matrix, $\lambda_2 = \lambda_2(\mathcal{L})$):

$$\frac{\lambda_2}{2} \le h \le \sqrt{2 \lambda_2}$$

Die obere Schranke ist konstruktiv: Der Sweep-Cut über $D^{-1/2} v$ erreicht $\min_k \varphi(S_k) \le \sqrt{2 \lambda_2}$, also gilt $h \le \min_k \varphi(S_k) \le \sqrt{2\lambda_2}$.

**Literatur.** M. Fiedler: *Algebraic connectivity of graphs*, Czechoslovak Mathematical Journal 23 (2), 1973, S. 298–305. J. Cheeger: *A lower bound for the smallest eigenvalue of the Laplacian*, in: Problems in Analysis, 1970, S. 195–199; für Graphen
N. Alon, V. D. Milman: *λ₁, isoperimetric inequalities for graphs, and superconcentrators*, Journal of Combinatorial Theory B 38 (1), 1985, S. 73–88. J. Shi, J. Malik: *Normalized cuts and image segmentation*, IEEE Transactions on Pattern Analysis and Machine Intelligence 22 (8), 2000, S. 888–905.
F. R. K. Chung: *Spectral Graph Theory*, AMS, 1997. Die Demo rechnet die Eigenzerlegung dicht (`numpy.linalg.eigh`), nicht mit Lanczos-Verfahren.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
