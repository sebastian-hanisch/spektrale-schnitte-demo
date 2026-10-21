"""Kennzahlen, Prüfung der Ergebnisse und Verdikte. Verdikte sind reine Funktionen (Art, Text): Zahlen kommen aus den Zeilen, nie aus dem Text.

Standorte sind in der Anzeige ab 1 nummeriert (`inst.name`), im Kern ab 0 (das Lehrbeispiel hat Buchstaben)."""

from dataclasses import dataclass, field

import numpy as np

import spk_algorithm as A
import spk_constants as C
import spk_scenario as S


@dataclass(frozen=True)
class Settings:
    kind: str = C.DEFAULT_KIND
    n: int = C.DEFAULT_N
    deg: float = C.DEFAULT_DEG
    links: int = C.DEFAULT_LINKS
    seed: int = C.DEFAULT_SEED
    lap: str = C.DEFAULT_LAP
    measure: str = C.DEFAULT_MEASURE


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    inst: S.Instance = field(compare=False, repr=False)
    exact: A.Exact | None = field(compare=False, repr=False)         # nur bei höchstens BRUTE_N Standorten
    mincut: int = field(compare=False)
    mincut_side: tuple = field(compare=False, repr=False)
    spectra: dict = field(compare=False, repr=False)                 # {"norm": Spectrum, "plain": Spectrum}
    sweeps: dict = field(compare=False, repr=False)                  # {"norm": Sweep, "plain": Sweep}

    @property
    def n(self):
        return self.inst.n

    @property
    def m(self):
        return self.inst.m

    @property
    def spec(self):
        return self.spectra[self.settings.lap]

    @property
    def sw(self):
        return self.sweeps[self.settings.lap]

    @property
    def best_k(self):
        return self.sw.best(self.settings.measure)

    @property
    def side(self):
        """Der beste Anfang des Sweeps für das gewählte Maß (Menge der Standorte, aufsteigend)."""
        return self.sw.prefix(self.best_k)

    @property
    def side_measures(self):
        """(Wert, RatioCut, Ncut, Konduktanz) des gewählten Schnitts, nachgezählt."""
        return A.measures(self.inst.adj, self.side)

    @property
    def sweep_value(self):
        return self.sw.series(self.settings.measure)[self.best_k - 1]

    @property
    def optimum(self):
        return self.exact.opt[self.settings.measure].value if self.exact else None

    @property
    def h(self):
        return self.exact.opt["cond"].value if self.exact else None

    @property
    def mincut_trivial(self):
        return A.is_trivial(self.n, set(self.mincut_side))

    @property
    def cheeger(self):
        """(untere Schranke, obere Schranke) aus λ₂ der normalisierten Matrix."""
        return A.cheeger_bounds(self.spectra["norm"].lam2)

    @property
    def median_k(self):
        return self.n // 2


def analyse(settings: Settings) -> Analysis:
    inst = S.make_instance(settings.kind, settings.n, settings.deg, settings.links, settings.seed)
    adj = inst.adj
    exact = A.exact_optima(adj) if inst.n <= C.BRUTE_N else None
    mc, side = A.min_cut(adj)
    spectra = {lap: A.spectrum(adj, lap) for lap in C.LAPS}
    sweeps = {lap: A.sweep(adj, spectra[lap].x) for lap in C.LAPS}
    return Analysis(settings, inst, exact, mc, side, spectra, sweeps)


def _close(a, b):
    return abs(a - b) <= C.TOL * max(1.0, abs(a), abs(b))


def validate(a: Analysis) -> list:
    """Unabhängige Prüfung vor der Anzeige: gibt die Liste der Probleme zurück (leer = alles in Ordnung)."""
    problems = []
    adj = a.inst.adj
    n = a.n
    for lap, spec in a.spectra.items():
        lap_name = "normalisiert" if lap == "norm" else "unnormalisiert"
        mat = A.laplacian(adj, lap)
        vec = np.array(spec.vec)
        if spec.lam2 <= C.TOL:
            problems.append(f"Die {lap_name}e Matrix hat λ₂ = 0: das Netz ist nicht zusammenhängend")
        if float(np.linalg.norm(mat @ vec - spec.lam2 * vec)) > 1e-8:
            problems.append(f"Die {lap_name}e Matrix: der Fiedler-Vektor erfüllt die Eigenwertgleichung nicht")
        sw = a.sweeps[lap]
        for k in range(1, n):
            c, r, nc, co = A.measures(adj, sw.prefix(k))
            if not (c == sw.cut[k - 1] and _close(r, sw.ratio[k - 1]) and _close(nc, sw.ncut[k - 1]) and _close(co, sw.cond[k - 1])):
                problems.append(f"Der Sweep ({lap_name}): die Werte des Anfangs der Länge {k} stimmen nicht mit der Nachzählung überein")
                break
    if a.exact is not None:
        lo, hi = a.cheeger
        h = a.h
        if not (lo <= h + C.TOL and h <= hi + C.TOL):
            problems.append("Die Schranken von Cheeger gelten nicht")
        if min(a.sweeps["norm"].cond) < h - C.TOL or min(a.sweeps["norm"].cond) > hi + C.TOL:
            problems.append("Der Sweep-Cut liegt nicht zwischen h und √(2 λ₂)")
        for key in C.OBJECTIVES:
            o = a.exact.opt[key]
            c, r, nc, co = A.measures(adj, o.side)
            if not _close({"cut": c, "ratio": r, "ncut": nc, "cond": co}[key], o.value):
                problems.append("Ein bester Schnitt der Aufzählung hat nicht den gemeldeten Wert")
                break
        if a.mincut != a.exact.opt["cut"].value:
            problems.append("Der kleinste Schnitt nach Stoer–Wagner weicht von der Aufzählung ab")
    if a.mincut > min(len(x) for x in adj):
        problems.append("Der kleinste Schnitt ist größer als der kleinste Grad")
    return problems


# --- Formate ---------------------------------------------------------------------------------------------------------------------------------------


def _i(x) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{x:,.0f}".replace(",", ".")


def _dec(x, digits=1) -> str:
    """Zahl mit Komma; ganze Werte ohne Nachkommastelle."""
    x = float(x)
    if x == int(x):
        return _i(x)
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _v(x) -> str:
    """Wert eines Schnittmaßes (drei Nachkommastellen, ganze Werte ohne)."""
    return _dec(x, 3)


def _pct(share, digits=0) -> str:
    return _dec(round(100.0 * share, digits), digits) if digits else _i(round(100.0 * share))


def _of(k, n) -> str:
    return f"{_i(k)} von {_i(n)}"


# --- Verdikte -------------------------------------------------------------------------------------------------------------------------------------


def _leitungen(c) -> str:
    return "1 Leitung" if c == 1 else f"{_i(c)} Leitungen"


def verdict_question(a: Analysis):
    """Kernfrage: kleinster Schnitt ohne Balance, was das gewählte Maß wählt und wie nah der Sweep-Cut am Optimum liegt."""
    s = a.settings
    n = a.n
    size = min(len(a.mincut_side), n - len(a.mincut_side))
    head = f"**Bei {_i(n)} Standorten kostet der kleinste Schnitt {_i(a.mincut)}"
    if a.exact and a.exact.opt["cut"].count > 1:
        o = a.exact.opt["cut"]
        head += f"; {_i(o.count)} Schnitte haben diesen Wert, " + ("alle trennen nur einen einzelnen Standort ab" if o.all_trivial else ("einer davon trennt nur einen einzelnen Standort ab" if o.any_trivial else "keiner trennt nur einen einzelnen Standort ab"))
    else:
        head += ", er trennt nur einen einzelnen Standort ab" if a.mincut_trivial else f", er trennt {_i(size)} von {_i(n)} Standorten ab"
    c = a.side_measures[0]
    head += f"; der Sweep-Cut mit {C.MEASURE_SHORT[s.measure]} wählt dagegen einen Schnitt mit {_leitungen(c)} und {_i(a.best_k)} gegen {_i(n - a.best_k)} Standorten.**"
    found = f" {C.MEASURE_SHORT[s.measure]} = {_v(a.sweep_value)}"
    if a.exact:
        opt = a.optimum
        found += ", das Optimum der Aufzählung" if _close(a.sweep_value, opt) else f", das {_dec(a.sweep_value / opt, 2)}-Fache des Optimums ({_v(opt)})"
    else:
        found += " (ohne Aufzählung keine Gegenprobe)"
    found += "."
    lo, hi = a.cheeger
    cheeger = f" Cheeger: λ₂ = {_v(a.spectra['norm'].lam2)} (normalisiert), also liegt die kleinste Konduktanz h zwischen {_v(lo)} und {_v(hi)}"
    if a.exact:
        cheeger += f"; sie ist {_v(a.h)}"
    cheeger += "."
    if a.spec.multiplicity > 1:
        cheeger += f" λ₂ ist {_i(a.spec.multiplicity)}-fach: der Fiedler-Vektor ist nicht eindeutig."
    return "info", head + found + cheeger


def verdict_exact(rows):
    nets, checks, wrong = (sum(r[k] for r in rows) for k in ("nets", "checks", "wrong"))
    n = rows[0]["n"]
    if wrong:
        return "warning", f"Auf {_i(nets)} Netzen weichen {_i(wrong)} von {_i(checks)} Prüfungen ab."
    return "info", (f"Auf {_i(nets)} Netzen mit {_i(n)} Standorten ({len(rows)} Netztypen) bestehen {_i(checks)} Prüfungen: λ₂ ist positiv, der Fiedler-Vektor erfüllt die Eigenwertgleichung und steht senkrecht auf dem Eigenvektor "
                    f"zum Eigenwert 0 (beide Matrizen), die Schranken von Cheeger gelten, der Sweep-Cut liegt zwischen h und √(2 λ₂), jeder Anfang des Sweeps stimmt mit der Nachzählung überein, jeder Sweep liegt nie "
                    f"unter dem Optimum der Aufzählung, und jeder beste Schnitt der Aufzählung hat den gemeldeten Wert. Abweichungen: 0.")


def verdict_objective(rows):
    nets = sum(r["nets"] for r in rows)
    s = {k: sum(r[k] for r in rows) for k in ("cut_all_trivial", "cut_any_trivial", "ncut_trivial", "ratio_trivial", "cond_trivial", "same_ncut_ratio", "same_ncut_cond")}
    text = (f"Der kleinste Schnitt besteht in {_of(s['cut_all_trivial'], nets)} Netzen nur aus einzelnen Standorten (in {_of(s['cut_any_trivial'], nets)} mindestens einer von ihnen); der beste Ncut-Schnitt ist es in "
            f"{_of(s['ncut_trivial'], nets)} Netzen, der beste RatioCut-Schnitt in {_of(s['ratio_trivial'], nets)} und der mit der kleinsten Konduktanz in {_of(s['cond_trivial'], nets)}. Ncut und RatioCut haben "
            f"dieselben besten Schnitte in {_of(s['same_ncut_ratio'], nets)} Netzen, Ncut und Konduktanz in {_of(s['same_ncut_cond'], nets)}.")
    return "info", text


def verdict_cheeger(rows):
    nets = sum(r["nets"] for r in rows)
    bad = sum(r["violations"] for r in rows)
    lo_min, lo_max = min(r["lo_min"] for r in rows), max(r["lo_max"] for r in rows)
    hi_min, hi_max = min(r["hi_min"] for r in rows), max(r["hi_max"] for r in rows)
    if bad:
        return "warning", f"Die Schranken von Cheeger sind in {_i(bad)} {'Fall' if bad == 1 else 'Fällen'} verletzt."
    return "info", (f"Die Schranken von Cheeger gelten in allen {_i(nets)} Netzen (Verletzungen: 0). Die kleinste Konduktanz h liegt das {_dec(lo_min)}- bis {_dec(lo_max)}-Fache über der unteren Schranke λ₂/2 und bei "
                    f"{_pct(hi_min)} bis {_pct(hi_max)} % der oberen Schranke √(2 λ₂): keine der beiden ist in diesen Netzen eng.")


def verdict_sweep(rows):
    nets = sum(r["nets"] for r in rows)
    s = {k: sum(r[k] for r in rows) for k in ("hit_cond", "hit_cond_plain", "hit_ncut", "hit_ratio_plain", "multiple")}
    worst = max(r["worst_cond"] for r in rows)
    text = (f"Der Sweep-Cut trifft die kleinste Konduktanz in {_of(s['hit_cond'], nets)} Netzen mit der normalisierten Matrix und in {_of(s['hit_cond_plain'], nets)} mit der unnormalisierten, den kleinsten Ncut "
            f"(normalisiert) in {_of(s['hit_ncut'], nets)} und den kleinsten RatioCut (unnormalisiert) in {_of(s['hit_ratio_plain'], nets)}; auf diesen Netzen liegt seine Konduktanz höchstens das {_dec(worst)}-Fache über h.")
    if s["multiple"]:
        text += f" In {_of(s['multiple'], nets)} Netzen ist λ₂ mehrfach (Gitter): dort hängt der Sweep von der Wahl des Fiedler-Vektors im Eigenraum ab."
    return "info", text


def verdict_big(rows):
    nets = sum(r["nets"] for r in rows)
    n = rows[0]["n"]
    phi_sweep = sum(r["sum_phi_sweep"] for r in rows) / nets
    phi_median = sum(r["sum_phi_median"] for r in rows) / nets
    better = sum(r["better"] for r in rows)
    small = sum(r["sum_small"] for r in rows) / nets
    ratios = {r["label"]: r["sum_phi_median"] / r["sum_phi_sweep"] for r in rows}
    top = max(ratios, key=ratios.get)
    text = (f"Bei {_i(n)} Standorten ({_i(nets)} Netze) liegt die Konduktanz des besten Sweep-Anfangs im Mittel bei {_v(phi_sweep)}, die der Halbierung am Median des Fiedler-Vektors bei {_v(phi_median)}; "
            f"in {_of(better, nets)} Netzen ist der beste Anfang nicht die Hälfte. Die kleinere Seite hat im Mittel {_pct(small)} % der Standorte. Der größte Unterschied: {top} (die Halbierung hat im Mittel das "
            f"{_dec(ratios[top])}-Fache der Konduktanz des Sweeps).")
    return "info", text
